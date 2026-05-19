
#include "pch.h"

#include "vk_ray/SBT.h"
#include "vk_ray/device.h"
#include "vk_ray/buffer.h"

// FORWARD DECLARATIONS ================================================================================================


namespace vr {

    // CONSTANTS =======================================================================================================

    // MACROS ==========================================================================================================

    // TYPES ===========================================================================================================

    // STATIC VARIABLES ================================================================================================

    // FUNCTION IMPLEMENTATION =========================================================================================

    // CLASS IMPLEMENTATION ============================================================================================

    // CLASS PUBLIC ====================================================================================================

    std::vector<uint8_t> device::get_handles_for_sbtbuffer(vk::Pipeline pipeline, uint32_t firstGroup, uint32_t groupCount) {

        uint32_t alignedSize = align_up(m_ray_tracing_properties.shaderGroupHandleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);
        uint32_t size = alignedSize * groupCount;
        std::vector<uint8_t> handles(size);

        auto result = m_device.getRayTracingShaderGroupHandlesKHR(pipeline, firstGroup, groupCount, size, handles.data(), m_dyn_loader);
        if (result != vk::Result::eSuccess)
            LOG(error, "get_handles_for_sbtbuffer: Failed to get ray tracing shader group handles");
        return handles;
    }


    void device::get_handles_for_sbtbuffer(vk::Pipeline pipeline, uint32_t firstGroup, uint32_t groupCount, void* data) {

        uint32_t alignedSize = align_up(m_ray_tracing_properties.shaderGroupHandleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);
        uint32_t size = alignedSize * groupCount;

        auto result = m_device.getRayTracingShaderGroupHandlesKHR(pipeline, firstGroup, groupCount, size, data, m_dyn_loader);
        if (result != vk::Result::eSuccess)
            LOG(error, "get_handles_for_sbtbuffer: Failed to get ray tracing shader group handles");
    }


    void device::write_to_sbt(sbt_buffer sbt_buffer, shader_group group, uint32_t groupIndex, void* data, uint32_t dataSize, void* mappedData) {

        allocated_buffer* buffer = nullptr;
        vk::StridedDeviceAddressRegionKHR* addressRegion;
        switch (group) {

            case shader_group::ray_gen:
                buffer = &sbt_buffer.ray_gen_buffer;
                addressRegion = &sbt_buffer.ray_gen_region;
                break;
            case shader_group::miss:
                buffer = &sbt_buffer.miss_buffer;
                addressRegion = &sbt_buffer.miss_region;
                break;
            case shader_group::hit_group:
                buffer = &sbt_buffer.hit_group_buffer;
                addressRegion = &sbt_buffer.hit_group_region;
                break;
            case shader_group::callable:
                buffer = &sbt_buffer.callable_buffer;
                addressRegion = &sbt_buffer.callable_region;
                break;
        }
        if (!buffer) {

            LOG(error, "WriteToSBT: Invalid shader group");
            return;
        }

        // Offset to the start of the requested group and apply the opaque handle size for the SBT
        uint32_t offset = (groupIndex * addressRegion->stride) + m_ray_tracing_properties.shaderGroupHandleSize;

        // make sure the data size is not too large
        if (offset + dataSize > addressRegion->size) {

            LOG(error, "WriteToSBT: Data size is too large for shader group");
            return;
        }

        if (mappedData)
            memcpy((uint8_t*)mappedData + offset, data, dataSize);     // if we have a mapped buffer, just copy the data
        else
            update_buffer(*buffer, data, dataSize, offset);              // else we update the buffer with the data
    }


    sbt_buffer device::create_sbt(vk::Pipeline pipeline, const sbt_info& sbt) {

        sbt_buffer outSBT;
        const uint32_t rgen_count = sbt.ray_gen_indices.size();
        const uint32_t miss_count = sbt.miss_indices.size();
        const uint32_t hit_count = sbt.hit_group_indices.size();
        const uint32_t call_count = sbt.callable_indices.size();
        // const uint32_t groupsCount = rgen_count + miss_count + hit_count + call_count;
        const uint32_t handleSize = m_ray_tracing_properties.shaderGroupHandleSize;

        uint32_t rgen_size = align_up(sbt.ray_gen_shader_record_size + handleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);
        uint32_t miss_size = align_up(sbt.miss_shader_record_size + handleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);
        uint32_t hit_size = align_up(sbt.hit_group_record_size + handleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);
        uint32_t call_size = align_up(sbt.callable_shader_record_size + handleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);

        // Create all buffers for the SBT
        if (sbt.ray_gen_indices.size() || sbt.reserve_ray_gen_groups)
            outSBT.ray_gen_buffer = create_buffer(rgen_size * (rgen_count + sbt.reserve_ray_gen_groups),
                vk::BufferUsageFlagBits::eShaderDeviceAddressKHR | vk::BufferUsageFlagBits::eShaderBindingTableKHR,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, m_ray_tracing_properties.shaderGroupBaseAlignment);

        if (sbt.miss_indices.size() || sbt.reserve_miss_groups)
            outSBT.miss_buffer = create_buffer(miss_size * (miss_count + sbt.reserve_miss_groups),
                vk::BufferUsageFlagBits::eShaderDeviceAddressKHR | vk::BufferUsageFlagBits::eShaderBindingTableKHR,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, m_ray_tracing_properties.shaderGroupBaseAlignment);

        if (sbt.hit_group_indices.size() || sbt.reserve_hit_groups)
            outSBT.hit_group_buffer = create_buffer(hit_size * (hit_count + sbt.reserve_hit_groups),
                vk::BufferUsageFlagBits::eShaderDeviceAddressKHR | vk::BufferUsageFlagBits::eShaderBindingTableKHR,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, m_ray_tracing_properties.shaderGroupBaseAlignment);

        if (sbt.callable_indices.size() || sbt.reserve_callable_groups)
            outSBT.callable_buffer = create_buffer(call_size * (call_count + sbt.reserve_callable_groups),
                vk::BufferUsageFlagBits::eShaderDeviceAddressKHR | vk::BufferUsageFlagBits::eShaderBindingTableKHR,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, m_ray_tracing_properties.shaderGroupBaseAlignment);

        // For filling the stride and size of the regions, we don't want to set stride when there is no shader of that
        // type. We didn't do this earlier because we needed to know the size of the shader group handles to reserve
        // space for them.
        if (rgen_count == 0)        rgen_size = 0;
        if (miss_count == 0)        miss_size = 0;
        if (hit_count == 0)         hit_size = 0;
        if (call_count == 0)        call_size = 0;

        // fill in offsets for all shader groups
        if (rgen_count > 0)
            outSBT.ray_gen_region = vk::StridedDeviceAddressRegionKHR()
                .setDeviceAddress(outSBT.ray_gen_buffer.dev_address)
                .setStride(rgen_size)
                .setSize(rgen_size * rgen_count);

        if (miss_count > 0)
            outSBT.miss_region = vk::StridedDeviceAddressRegionKHR()
                .setDeviceAddress(outSBT.miss_buffer.dev_address)
                .setStride(miss_size)
                .setSize(miss_size * miss_count);

        if (hit_count > 0)
            outSBT.hit_group_region = vk::StridedDeviceAddressRegionKHR()
                .setDeviceAddress(outSBT.hit_group_buffer.dev_address)
                .setStride(hit_size)
                .setSize(hit_size * hit_count);

        if (call_count > 0)
            outSBT.callable_region = vk::StridedDeviceAddressRegionKHR()
                .setDeviceAddress(outSBT.callable_buffer.dev_address)
                .setStride(call_size)
                .setSize(call_size * call_count);

        // const uint8_t* opaqueHandle = new uint8_t[handleSize];

        // copy records and shader handles into the SBT buffer
        uint8_t *rgen_data = rgen_size > 0 ? (uint8_t *)map_buffer(outSBT.ray_gen_buffer) : nullptr;
        for (uint32_t i = 0; rgen_data && i < rgen_count; i++)
        {
            uint32_t shader_index = sbt.ray_gen_indices[i];
            const uint8_t* dest = rgen_data + (i * rgen_size);
            get_handles_for_sbtbuffer(pipeline, shader_index, 1, (void *)dest);
            rgen_data += rgen_size; // advance to the next record
        }
        uint8_t *miss_data = miss_size > 0 ? (uint8_t *)map_buffer(outSBT.miss_buffer) : nullptr;
        for (uint32_t i = 0; miss_data && i < miss_count; i++)
        {
            uint32_t shader_index = sbt.miss_indices[i];
            const uint8_t* dest = miss_data + (i * miss_size);
            get_handles_for_sbtbuffer(pipeline, shader_index, 1, (void *)dest);
        }
        uint8_t *hit_data = hit_size > 0 ? (uint8_t *)map_buffer(outSBT.hit_group_buffer) : nullptr;
        for (uint32_t i = 0; hit_data && i < hit_count; i++)
        {
            uint32_t shader_index = sbt.hit_group_indices[i];
            const uint8_t* dest = hit_data + (i * hit_size);
            get_handles_for_sbtbuffer(pipeline, shader_index, 1, (void *)dest);
        }
        uint8_t *call_data = call_size > 0 ? (uint8_t *)map_buffer(outSBT.callable_buffer) : nullptr;
        for (uint32_t i = 0; call_data && i < call_count; i++)
        {
            uint32_t shader_index = sbt.callable_indices[i];
            const uint8_t* dest = call_data + (i * call_size);
            get_handles_for_sbtbuffer(pipeline, shader_index, 1, (void *)dest);
        }

        // unmap all the buffers
        if (rgen_data)   unmap_buffer(outSBT.ray_gen_buffer);
        if (miss_data)   unmap_buffer(outSBT.miss_buffer);
        if (hit_data)    unmap_buffer(outSBT.hit_group_buffer);
        if (call_data)   unmap_buffer(outSBT.callable_buffer);

        return outSBT;
    }


    bool device::rebuild_sbt(vk::Pipeline pipeline, sbt_buffer& buffer, const sbt_info& sbt) {

        if (!can_sbt_fit_shaders(buffer, sbt))
            return false;

        const uint32_t rgen_count = sbt.ray_gen_indices.size();
        const uint32_t miss_count = sbt.miss_indices.size();
        const uint32_t hit_count = sbt.hit_group_indices.size();
        const uint32_t call_count = sbt.callable_indices.size();
        // const uint32_t groupsCount = rgen_count + miss_count + hit_count + call_count;

        const uint32_t handleSize = align_up(m_ray_tracing_properties.shaderGroupHandleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);
        const uint32_t rgen_size = align_up(sbt.ray_gen_shader_record_size + handleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);
        const uint32_t miss_size = align_up(sbt.miss_shader_record_size + handleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);
        const uint32_t hit_size = align_up(sbt.hit_group_record_size + handleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);
        const uint32_t call_size = align_up(sbt.callable_shader_record_size + handleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);

        // we have to rewrite opaque handles to all the groups in the SBT, because on some implementations just keeping
        // the old opaque handles and adding new opaque handles to the new added groups doesn't work

        // const uint8_t* opaqueHandle = new uint8_t[handleSize];

        // copy records and shader handles into the SBT buffer
        uint8_t *rgen_data = rgen_size > 0 ? (uint8_t *)map_buffer(buffer.ray_gen_buffer) : nullptr;
        for (uint32_t i = 0; rgen_data && i < rgen_count; i++)
        {
            uint32_t shader_index = sbt.ray_gen_indices[i];
            const uint8_t* dest = rgen_data + (i * rgen_size);
            get_handles_for_sbtbuffer(pipeline, shader_index, 1, (void *)dest);
            rgen_data += rgen_size; // advance to the next record
        }

        uint8_t* miss_data = miss_size > 0 ? (uint8_t *)map_buffer(buffer.miss_buffer) : nullptr;
        for (uint32_t i = 0; miss_data && i < miss_count; i++)
        {
            uint32_t shader_index = sbt.miss_indices[i];
            const uint8_t* dest = miss_data + (i * miss_size);
            get_handles_for_sbtbuffer(pipeline, shader_index, 1, (void *)dest);
        }

        uint8_t* hit_data = hit_size > 0 ? (uint8_t *)map_buffer(buffer.hit_group_buffer) : nullptr;
        for (uint32_t i = 0; hit_data && i < hit_count; i++)
        {
            uint32_t shader_index = sbt.hit_group_indices[i];
            const uint8_t* dest = hit_data + (i * hit_size);
            get_handles_for_sbtbuffer(pipeline, shader_index, 1, (void *)dest);
        }

        uint8_t* call_data = call_size > 0 ? (uint8_t *)map_buffer(buffer.callable_buffer) : nullptr;
        for (uint32_t i = 0; call_data && i < call_count; i++)
        {
            uint32_t shader_index = sbt.callable_indices[i];
            const uint8_t* dest = call_data + (i * call_size);
            get_handles_for_sbtbuffer(pipeline, shader_index, 1, (void *)dest);
        }

        // unmap all the buffers
        if (rgen_data)   unmap_buffer(buffer.ray_gen_buffer);
        if (miss_data)   unmap_buffer(buffer.miss_buffer);
        if (hit_data)    unmap_buffer(buffer.hit_group_buffer);
        if (call_data)   unmap_buffer(buffer.callable_buffer);

        // Some groups may have gotten additional shaders, so we need to update the stride and size of the regions
        // We don't have to worry about the buffer sizes if they don't fit as it is already checked at the beginning of
        // this function

        if (rgen_count > 0)
            buffer.ray_gen_region = vk::StridedDeviceAddressRegionKHR()
                .setDeviceAddress(buffer.ray_gen_buffer.dev_address)
                .setStride(rgen_size)
                .setSize(rgen_size * rgen_count);

        if (miss_count > 0)
            buffer.miss_region = vk::StridedDeviceAddressRegionKHR()
                .setDeviceAddress(buffer.miss_buffer.dev_address)
                .setStride(miss_size)
                .setSize(miss_size * miss_count);

        if (hit_count > 0)
            buffer.hit_group_region = vk::StridedDeviceAddressRegionKHR()
                .setDeviceAddress(buffer.hit_group_buffer.dev_address)
                .setStride(hit_size)
                .setSize(hit_size * hit_count);

        if (call_count > 0)
            buffer.callable_region = vk::StridedDeviceAddressRegionKHR()
                .setDeviceAddress(buffer.callable_buffer.dev_address)
                .setStride(call_size)
                .setSize(call_size * call_count);

        return true;
    }

    void device::copy_sbt(sbt_buffer& source, sbt_buffer& destination)
    {
        uint8_t *dstRgenData = destination.ray_gen_region.size > 0 ? (uint8_t *)map_buffer(destination.ray_gen_buffer) : nullptr;
        uint8_t *dstMissData = destination.miss_region.size > 0 ? (uint8_t *)map_buffer(destination.miss_buffer) : nullptr;
        uint8_t *dstHitData = destination.hit_group_region.size > 0 ? (uint8_t *)map_buffer(destination.hit_group_buffer) : nullptr;
        uint8_t *dstCallData = destination.callable_region.size > 0 ? (uint8_t *)map_buffer(destination.callable_buffer) : nullptr;

        uint8_t *srcRgenData = source.ray_gen_region.size > 0 ? (uint8_t *)map_buffer(source.ray_gen_buffer) : nullptr;
        uint8_t *srcMissData = source.miss_region.size > 0 ? (uint8_t *)map_buffer(source.miss_buffer) : nullptr;
        uint8_t *srcHitData = source.hit_group_region.size > 0 ? (uint8_t *)map_buffer(source.hit_group_buffer) : nullptr;
        uint8_t *srcCallData = source.callable_region.size > 0 ? (uint8_t *)map_buffer(source.callable_buffer) : nullptr;

        if (dstRgenData && srcRgenData) memcpy(dstRgenData, srcRgenData, source.ray_gen_region.size);
        if (dstMissData && srcMissData) memcpy(dstMissData, srcMissData, source.miss_region.size);
        if (dstHitData && srcHitData)   memcpy(dstHitData, srcHitData, source.hit_group_region.size);
        if (dstCallData && srcCallData) memcpy(dstCallData, srcCallData, source.callable_region.size);

        if (dstRgenData)    unmap_buffer(destination.ray_gen_buffer);
        if (dstMissData)    unmap_buffer(destination.miss_buffer);
        if (dstHitData)     unmap_buffer(destination.hit_group_buffer);
        if (dstCallData)    unmap_buffer(destination.callable_buffer);

        if (srcRgenData)    unmap_buffer(source.ray_gen_buffer);
        if (srcMissData)    unmap_buffer(source.miss_buffer);
        if (srcHitData)     unmap_buffer(source.hit_group_buffer);
        if (srcCallData)    unmap_buffer(source.callable_buffer);
    }


    bool device::can_sbt_fit_shaders(sbt_buffer& buffer, const sbt_info& sbt_info) {

        bool extendable = true;

        const uint32_t rgen_size = align_up(sbt_info.ray_gen_shader_record_size + m_ray_tracing_properties.shaderGroupHandleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);
        const uint32_t miss_size = align_up(sbt_info.miss_shader_record_size + m_ray_tracing_properties.shaderGroupHandleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);
        const uint32_t hit_size = align_up(sbt_info.hit_group_record_size + m_ray_tracing_properties.shaderGroupHandleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);
        const uint32_t call_size = align_up(sbt_info.callable_shader_record_size + m_ray_tracing_properties.shaderGroupHandleSize, m_ray_tracing_properties.shaderGroupHandleAlignment);

        const uint32_t rgen_bytes_needed = sbt_info.ray_gen_indices.size() * rgen_size;
        const uint32_t miss_bytes_needed = sbt_info.miss_indices.size() * miss_size;
        const uint32_t hit_bytes_needed = sbt_info.hit_group_indices.size() * hit_size;
        const uint32_t call_bytes_needed = sbt_info.callable_indices.size() * call_size;

        if (rgen_bytes_needed > buffer.ray_gen_buffer.size)     return false;
        if (miss_bytes_needed > buffer.miss_buffer.size)        return false;
        if (hit_bytes_needed > buffer.hit_group_buffer.size)    return false;
        if (call_bytes_needed > buffer.callable_buffer.size)    return false;

        return extendable;
    }


    void device::destroy_sbt_buffer(sbt_buffer& buffer){

        // destroy all the buffers if they were created
        if (buffer.ray_gen_buffer.buffer)       destroy_buffer(buffer.ray_gen_buffer);
        if (buffer.miss_buffer.buffer)          destroy_buffer(buffer.miss_buffer);
        if (buffer.hit_group_buffer.buffer)     destroy_buffer(buffer.hit_group_buffer);
        if (buffer.callable_buffer.buffer)      destroy_buffer(buffer.callable_buffer);

        buffer.ray_gen_region = vk::StridedDeviceAddressRegionKHR();
        buffer.miss_region = vk::StridedDeviceAddressRegionKHR();
        buffer.hit_group_region = vk::StridedDeviceAddressRegionKHR();
        buffer.callable_region = vk::StridedDeviceAddressRegionKHR();
    }

    // CLASS PROTECTED =================================================================================================

    // CLASS PRIVATE ===================================================================================================

}
