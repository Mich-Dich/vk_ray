
#include "pch.h"

#include "vk_ray/accel_struct.h"
#include "vk_ray/device.h"

namespace vr {

    std::pair<blas_handle, blas_build_info> device::create_blas(const blas_create_info& info) {

        blas_build_info out_build_info = {};
        blas_handle out_accel = {};
        uint32_t geom_size = info.geometries.size();

        std::vector<uint32_t> max_primitive_counts(geom_size);

        // create a unique pointer to the array of geometries
        out_build_info.geometries = std::make_unique_for_overwrite<vk::AccelerationStructureGeometryKHR[]>(geom_size);
        out_build_info.geometry_count = geom_size;

        out_build_info.ranges = std::make_unique_for_overwrite<vk::AccelerationStructureBuildRangeInfoKHR[]>(geom_size);
        out_build_info.ranges_count = geom_size;

        for (size_t i = 0; i < geom_size; i++) {

            // Convert the geometries to the vulkan format
            out_build_info.geometries[i] = vk::AccelerationStructureGeometryKHR()
                .setGeometry(convert_to_vulkan_geometry(info.geometries[i]))
                .setFlags(info.geometries[i].flags)
                .setGeometryType(info.geometries[i].type);

            // Fill in the range info
            out_build_info.ranges[i] = vk::AccelerationStructureBuildRangeInfoKHR()
                .setFirstVertex(0)
                .setPrimitiveCount(info.geometries[i].primitive_count)
                .setPrimitiveOffset(0)
                .setTransformOffset(0);

            max_primitive_counts[i] = info.geometries[i].primitive_count;
        }

        // Create the build info
        out_build_info.build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR()
            .setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
            .setFlags(info.flags)
            .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
            .setDstAccelerationStructure(nullptr)
            .setPGeometries(out_build_info.geometries.get())
            .setGeometryCount(out_build_info.geometry_count);

        // Get the size requirements for the acceleration structure
        m_device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice,
            &out_build_info.build_geometry_info, max_primitive_counts.data(), &out_build_info.build_sizes, m_dyn_loader); // This will fill in the size requirements

        // Create the buffer for the acceleration structure
        out_accel.buffer = create_buffer(out_build_info.build_sizes.accelerationStructureSize, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR, 0); // no flags for VMA

        // Create the acceleration structure
        auto create_info = vk::AccelerationStructureCreateInfoKHR()
            .setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
            .setBuffer(out_accel.buffer.buffer)
            .setSize(out_build_info.build_sizes.accelerationStructureSize);

        out_accel.acceleration_structure = m_device.createAccelerationStructureKHR(create_info, nullptr, m_dyn_loader);

        // Get the address of the acceleration structure, because it may vary from getBufferDeviceAddress(...)
        out_accel.buffer.dev_address = m_device.getAccelerationStructureAddressKHR(vk::AccelerationStructureDeviceAddressInfoKHR()
            .setAccelerationStructure(out_accel.acceleration_structure),
            m_dyn_loader);

        // Fill in the build info with the acceleration structure
        out_build_info.build_geometry_info.setDstAccelerationStructure(out_accel.acceleration_structure);

        return std::make_pair(out_accel, out_build_info);
    }


    void device::build_blas(const std::vector<blas_build_info>& build_infos, vk::CommandBuffer cmd_buf) {

        std::vector<vk::AccelerationStructureBuildRangeInfoKHR*> p_build_range_infos;
        std::vector<vk::AccelerationStructureBuildGeometryInfoKHR> build_geometry_infos;
        p_build_range_infos.reserve(build_infos.size());
        build_geometry_infos.reserve(build_infos.size());

        for (uint32_t i = 0; i < build_infos.size(); i++) {

            p_build_range_infos.push_back(build_infos[i].ranges.get());
            build_geometry_infos.push_back(build_infos[i].build_geometry_info);
        }

        // Build the acceleration structures
        cmd_buf.buildAccelerationStructuresKHR(build_infos.size(), build_geometry_infos.data(), p_build_range_infos.data(), m_dyn_loader);
    }


    blas_build_info device::update_blas(blas_update_info& update_info) {

        assert(update_info.new_geometry_addresses.size() == update_info.source_build_info.geometry_count &&
               "The number of new geometry addresses must match the number of geometries in the source build info");

        bool use_source_device_address = update_info.new_geometry_addresses.size() == 0;
        uint32_t geom_size = update_info.source_build_info.geometry_count;
        std::vector<uint32_t> max_primitive_counts(geom_size);

        blas_build_info out_build_info = update_info.source_build_info;
        out_build_info.build_geometry_info.setMode(vk::BuildAccelerationStructureModeKHR::eUpdate);

        // setup the primitive counts and ranges
        for (uint32_t i = 0; i < geom_size; i++) {

            if (!use_source_device_address) {

                auto geomType = update_info.source_build_info.geometries[i].geometryType;
                if (geomType == vk::GeometryTypeKHR::eTriangles) {

                    out_build_info.geometries[i].geometry.triangles.vertexData = update_info.new_geometry_addresses[i].vertex_dev_address;
                    out_build_info.geometries[i].geometry.triangles.indexData = update_info.new_geometry_addresses[i].index_dev_address;
                }
                else if (geomType == vk::GeometryTypeKHR::eAabbs) {

                    out_build_info.geometries[i].geometry.aabbs.data = update_info.new_geometry_addresses[i].vertex_dev_address;
                }
            }

            max_primitive_counts[i] = update_info.source_build_info.ranges[i].primitiveCount;
            out_build_info.ranges[i].primitiveOffset = 0;
            out_build_info.ranges[i].primitiveCount = update_info.source_build_info.ranges[i].primitiveCount;
        }

        // Get the size requirements for the acceleration structure
        m_device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice,
            &out_build_info.build_geometry_info, max_primitive_counts.data(),
            &out_build_info.build_sizes,
            m_dyn_loader); // This will fill in the size requirements

        // seup dst and src acceleration structures
        out_build_info.build_geometry_info.srcAccelerationStructure = update_info.source_blas->acceleration_structure;
        out_build_info.build_geometry_info.dstAccelerationStructure = update_info.source_blas->acceleration_structure;
        return out_build_info;
    }


    compaction_request device::request_compaction(const std::vector<blas_handle*>& source_blas) {

        compaction_request out_request = {};

        auto create_info = vk::QueryPoolCreateInfo()
            .setQueryType(vk::QueryType::eAccelerationStructureCompactedSizeKHR)
            .setQueryCount(source_blas.size());

        for (auto &blas : source_blas)
            out_request.source_blas.push_back(blas->acceleration_structure);

        out_request.compaction_query_pool = m_device.createQueryPool(create_info);

        return out_request;
    }

    std::vector<uint64_t> device::get_compaction_sizes(compaction_request &request, vk::CommandBuffer cmd_buf) {

        uint32_t blas_count = request.source_blas.size();
        auto [result, values] = m_device.getQueryPoolResults<uint64_t>(request.compaction_query_pool, 0, blas_count, sizeof(uint64_t) * blas_count, sizeof(uint64_t));

        if (result == vk::Result::eSuccess) {

            // Destroy the query pool, we don't need it anymore
            m_device.destroyQueryPool(request.compaction_query_pool);
            request.compaction_query_pool = nullptr;
            return values;
        }

        cmd_buf.resetQueryPool(request.compaction_query_pool, 0, blas_count);
        cmd_buf.writeAccelerationStructuresPropertiesKHR(request.source_blas, vk::QueryType::eAccelerationStructureCompactedSizeKHR,
            request.compaction_query_pool, 0, m_dyn_loader);

        return std::vector<uint64_t>(); // return empty vector
    }


    std::vector<blas_handle> device::compact_blas(compaction_request& request, const std::vector<uint64_t>& sizes, vk::CommandBuffer cmd_buf) {

        uint32_t blas_count = request.source_blas.size();
        std::vector<blas_handle> new_blas_to_return(blas_count);

        for (uint32_t i = 0; i < blas_count; i++) {

            if (sizes[i] == 0) continue;

            // Create buffer
            allocated_buffer compact_buffer = create_buffer(sizes[i], vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR, 0);

            // Create the compacted acceleration structure
            auto create_info = vk::AccelerationStructureCreateInfoKHR()
                .setSize(sizes[i])
                .setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
                .setBuffer(compact_buffer.buffer);

            auto compact_accel = m_device.createAccelerationStructureKHR(create_info, nullptr, m_dyn_loader);

            auto copyInfo = vk::CopyAccelerationStructureInfoKHR()
                .setSrc(request.source_blas[i])
                .setDst(compact_accel)
                .setMode(vk::CopyAccelerationStructureModeKHR::eCompact);

            cmd_buf.copyAccelerationStructureKHR(copyInfo, m_dyn_loader);

            // Set the new acceleration structure
            new_blas_to_return[i].acceleration_structure = compact_accel;
            new_blas_to_return[i].buffer = compact_buffer;
            auto address_info = vk::AccelerationStructureDeviceAddressInfoKHR().setAccelerationStructure(new_blas_to_return[i].acceleration_structure);
            new_blas_to_return[i].buffer.dev_address = m_device.getAccelerationStructureAddressKHR(address_info, m_dyn_loader);
        }
        return new_blas_to_return;
    }


    std::vector<blas_handle> device::compact_blas(compaction_request& request, const std::vector<uint64_t>& sizes,
        std::vector<blas_handle*> old_blas, vk::CommandBuffer cmd_buf) {

        uint32_t blas_count = request.source_blas.size();
        std::vector<blas_handle> old_blas_to_return(blas_count);

        for (uint32_t i = 0; i < blas_count; i++) {

            if (sizes[i] == 0) continue;

            // Create buffer
            allocated_buffer compact_buffer = create_buffer(sizes[i], vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR, 0);

            // Create the compacted acceleration structure
            auto create_info = vk::AccelerationStructureCreateInfoKHR()
                .setSize(sizes[i])
                .setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
                .setBuffer(compact_buffer.buffer);

            auto compact_accel = m_device.createAccelerationStructureKHR(create_info, nullptr, m_dyn_loader);

            auto copyInfo = vk::CopyAccelerationStructureInfoKHR()
                .setSrc(request.source_blas[i])
                .setDst(compact_accel)
                .setMode(vk::CopyAccelerationStructureModeKHR::eCompact);

            cmd_buf.copyAccelerationStructureKHR(copyInfo, m_dyn_loader);

            // store the old acceleration structure
            old_blas_to_return[i].acceleration_structure = old_blas[i]->acceleration_structure;
            old_blas_to_return[i].buffer = old_blas[i]->buffer;
            auto address_info = vk::AccelerationStructureDeviceAddressInfoKHR().setAccelerationStructure(old_blas_to_return[i].acceleration_structure);
            old_blas_to_return[i].buffer.dev_address = m_device.getAccelerationStructureAddressKHR(address_info, m_dyn_loader);

            // replace
            old_blas[i]->acceleration_structure = compact_accel;
            old_blas[i]->buffer = compact_buffer;
        }
        return old_blas_to_return;
    }


    allocated_buffer device::create_scratch_buffer_from_build_infos(std::vector<blas_build_info>& build_infos) {

        uint32_t scratch_size = get_scratch_buffer_size(build_infos);
        auto out_scratch_buffer = create_scratch_buffer(scratch_size);
        bind_scratch_buffer_to_build_infos(out_scratch_buffer, build_infos);

        return out_scratch_buffer;
    }


    allocated_buffer device::create_scratch_buffer_from_build_info(blas_build_info& build_info) {

        uint32_t scratch_size = build_info.build_geometry_info.mode == vk::BuildAccelerationStructureModeKHR::eBuild
            ? build_info.build_sizes.buildScratchSize
            : build_info.build_sizes.updateScratchSize;

        auto out_scratch_buffer = create_scratch_buffer(scratch_size);
        bind_scratch_adress_to_build_info(out_scratch_buffer.dev_address, build_info);
        return out_scratch_buffer;
    }


    void device::bind_scratch_adress_to_build_info(vk::DeviceAddress scratch_addr, blas_build_info& build_info) {

        build_info.build_geometry_info.setScratchData(align_up(scratch_addr, (uint64_t)m_accel_properties.minAccelerationStructureScratchOffsetAlignment));
    }


    uint32_t device::get_scratch_buffer_size(const std::vector<blas_build_info>& build_infos) {

        uint32_t scratch_size = 0;
        for (auto &info : build_infos) {

            vk::BuildAccelerationStructureModeKHR mode = info.build_geometry_info.mode;
            if (mode == vk::BuildAccelerationStructureModeKHR::eBuild)
                scratch_size += align_up(info.build_sizes.buildScratchSize, (uint64_t)m_accel_properties.minAccelerationStructureScratchOffsetAlignment);
            else
                scratch_size += align_up(info.build_sizes.updateScratchSize, (uint64_t)m_accel_properties.minAccelerationStructureScratchOffsetAlignment);
        }

        return scratch_size;
    }


    void device::bind_scratch_buffer_to_build_infos(const vr::allocated_buffer& buffer, std::vector<blas_build_info>& build_infos) {

        vk::DeviceAddress scratch_data_addr = buffer.dev_address;
        for (auto &info : build_infos)
        {
            vk::BuildAccelerationStructureModeKHR mode = info.build_geometry_info.mode;
            bind_scratch_adress_to_build_info(scratch_data_addr, info);
            if (mode == vk::BuildAccelerationStructureModeKHR::eBuild)
                scratch_data_addr += align_up(info.build_sizes.buildScratchSize, (uint64_t)m_accel_properties.minAccelerationStructureScratchOffsetAlignment);
            else
                scratch_data_addr += align_up(info.build_sizes.updateScratchSize, (uint64_t)m_accel_properties.minAccelerationStructureScratchOffsetAlignment);
        }
    }

    //--------------------------------------------------------------------------------------
    // TLAS FUCNTIONS
    //--------------------------------------------------------------------------------------

    std::pair<tlas_handle, tlas_build_info> device::create_tlas(const tlas_create_info &info) {

        tlas_handle out_accel = {};
        tlas_build_info out_build_info = {};

        auto build_geometry = vk::AccelerationStructureGeometryDataKHR().setInstances(
            vk::AccelerationStructureGeometryInstancesDataKHR().setArrayOfPointers(false));

        out_build_info.geometry = std::make_shared<vk::AccelerationStructureGeometryKHR>(vk::AccelerationStructureGeometryKHR()
            .setGeometry(build_geometry)
            .setGeometryType(vk::GeometryTypeKHR::eInstances)
            .setFlags(vk::GeometryFlagBitsKHR::eOpaque));

        // Create the build info
        out_build_info.build_geometry_info =
            vk::AccelerationStructureBuildGeometryInfoKHR()
                .setType(vk::AccelerationStructureTypeKHR::eTopLevel)
                .setFlags(info.flags)
                .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
                .setDstAccelerationStructure(nullptr)
                .setPGeometries(out_build_info.geometry.get())
                .setGeometryCount(
                    1); // MUST BE 1
                        // (https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureBuildGeometryInfoKHR.html#VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03790)

        // Fill Range Info
        out_build_info.range_info =
            vk::AccelerationStructureBuildRangeInfoKHR()
                .setPrimitiveCount(0) // number of instances, will be modified in the build function
                .setPrimitiveOffset(0)
                .setFirstVertex(0)
                .setTransformOffset(0);

        // Get the size requirements for the acceleration structure
        m_device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, &out_build_info.build_geometry_info,
            &info.max_instance_count, // max number of instances/primitives in the geometry
            &out_build_info.build_sizes, m_dyn_loader);

        // Create the buffer for the acceleration structure
        out_accel.buffer = create_buffer(out_build_info.build_sizes.accelerationStructureSize,
                                       vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR, 0);

        // Create the acceleration structure
        auto create_info = vk::AccelerationStructureCreateInfoKHR()
                              .setType(vk::AccelerationStructureTypeKHR::eTopLevel)
                              .setBuffer(out_accel.buffer.buffer)
                              .setSize(out_build_info.build_sizes.accelerationStructureSize);

        out_accel.acceleration_structure = m_device.createAccelerationStructureKHR(create_info, nullptr, m_dyn_loader);

        out_accel.buffer.dev_address = m_device.getAccelerationStructureAddressKHR(
            vk::AccelerationStructureDeviceAddressInfoKHR().setAccelerationStructure(out_accel.acceleration_structure),
            m_dyn_loader);

        // Fill in the build info with the acceleration structure

        out_build_info.build_geometry_info.setDstAccelerationStructure(out_accel.acceleration_structure);

        out_build_info.max_instance_count = info.max_instance_count;

        return std::make_pair(out_accel, out_build_info);
    }


    void device::build_tlas(tlas_build_info &build_info, const allocated_buffer &InstanceBuffer, uint32_t instanceCount, vk::CommandBuffer cmd_buf) {

        build_info.range_info.primitiveCount = instanceCount;
        build_info.geometry->geometry.instances.data = InstanceBuffer.dev_address;
        auto* p_build_range_info = &build_info.range_info;
        cmd_buf.buildAccelerationStructuresKHR(1, &build_info.build_geometry_info, &p_build_range_info, m_dyn_loader);
    }


    std::pair<tlas_handle, tlas_build_info> device::update_tlas(tlas_handle &old_tlas, tlas_build_info &old_build_info, bool destroy_old) {

        tlas_handle out_accel = old_tlas;
        tlas_build_info out_build_info = old_build_info;

        // Create new acceleration structure
        // sizes are the same as the old one, because we are only creating a new one with the same number of instances
        // otherwise we would need create a new one with the new number of instances, from CreateTLAS(...)

        auto create_info = vk::AccelerationStructureCreateInfoKHR()
            .setType(vk::AccelerationStructureTypeKHR::eTopLevel)
            .setBuffer(out_accel.buffer.buffer)
            .setSize(out_build_info.build_sizes.accelerationStructureSize);

        out_accel.acceleration_structure = m_device.createAccelerationStructureKHR(create_info, nullptr, m_dyn_loader);

        out_accel.buffer.dev_address = m_device.getAccelerationStructureAddressKHR(vk::AccelerationStructureDeviceAddressInfoKHR()
            .setAccelerationStructure(out_accel.acceleration_structure),
            m_dyn_loader);

        out_build_info.build_geometry_info.setDstAccelerationStructure(out_accel.acceleration_structure);

        // destroy the old one
        if (destroy_old)
            destroy_acceleration_structure(old_tlas.acceleration_structure);

        return std::make_pair(out_accel, out_build_info);
    }


    void device::bind_scratch_adress_to_build_info(vk::DeviceAddress scratch_addr, tlas_build_info& build_info) {

        build_info.build_geometry_info.setScratchData(align_up(scratch_addr, (uint64_t)m_accel_properties.minAccelerationStructureScratchOffsetAlignment));
    }


    void device::bind_scratch_buffer_to_build_infos(const vr::allocated_buffer& buffer, std::vector<tlas_build_info>& build_infos) {

        vk::DeviceAddress scratch_data_addr = buffer.dev_address;
        for (auto &info : build_infos)
        {
            vk::BuildAccelerationStructureModeKHR mode = info.build_geometry_info.mode;
            bind_scratch_adress_to_build_info(scratch_data_addr, info);
            if (mode == vk::BuildAccelerationStructureModeKHR::eBuild)
                scratch_data_addr += align_up(info.build_sizes.buildScratchSize, (uint64_t)m_accel_properties.minAccelerationStructureScratchOffsetAlignment);
            else
                scratch_data_addr += align_up(info.build_sizes.updateScratchSize, (uint64_t)m_accel_properties.minAccelerationStructureScratchOffsetAlignment);
        }
    }


    uint32_t device::get_scratch_buffer_size(const std::vector<tlas_build_info> &build_infos) {

        uint32_t scratch_size = 0;
        for (auto &info : build_infos) {

            vk::BuildAccelerationStructureModeKHR mode = info.build_geometry_info.mode;
            if (mode == vk::BuildAccelerationStructureModeKHR::eBuild)
                scratch_size += align_up(info.build_sizes.buildScratchSize, (uint64_t)m_accel_properties.minAccelerationStructureScratchOffsetAlignment);
            else
                scratch_size += align_up(info.build_sizes.updateScratchSize, (uint64_t)m_accel_properties.minAccelerationStructureScratchOffsetAlignment);
        }
        return scratch_size;
    }


    allocated_buffer device::create_scratch_buffer_from_build_infos(std::vector<tlas_build_info> &build_infos) {

        uint32_t scratch_size = get_scratch_buffer_size(build_infos);
        auto out_scratch_buffer = create_scratch_buffer(scratch_size);
        bind_scratch_buffer_to_build_infos(out_scratch_buffer, build_infos);
        return out_scratch_buffer;
    }


    allocated_buffer device::create_scratch_buffer_from_build_info(tlas_build_info& build_info) {

        uint32_t scratch_size = build_info.build_geometry_info.mode == vk::BuildAccelerationStructureModeKHR::eBuild
            ? build_info.build_sizes.buildScratchSize
            : build_info.build_sizes.updateScratchSize;

        auto out_scratch_buffer = create_scratch_buffer(scratch_size);

        bind_scratch_adress_to_build_info(out_scratch_buffer.dev_address, build_info);
        return out_scratch_buffer;
    }


    void device::add_acceleration_build_barrier(vk::CommandBuffer cmd_buf) {

        // accel build barrier for for next build
        auto barrier = vk::MemoryBarrier()
            .setSrcAccessMask(vk::AccessFlagBits::eAccelerationStructureWriteKHR)
            .setDstAccessMask(vk::AccessFlagBits::eAccelerationStructureReadKHR);

        cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, (vk::DependencyFlagBits)0, 1,
            &barrier, 0, nullptr, 0, nullptr);
    }


    void device::destroy_blas(std::vector<blas_handle> &blas) {

        for (auto &b : blas) {
            m_device.destroyAccelerationStructureKHR(b.acceleration_structure, nullptr, m_dyn_loader);
            destroy_buffer(b.buffer);
        }
    }


    void device::destroy_blas(blas_handle &blas)
    {
        m_device.destroyAccelerationStructureKHR(blas.acceleration_structure, nullptr, m_dyn_loader);
        destroy_buffer(blas.buffer);
    }


    void device::destroy_tlas(tlas_handle &tlas) {

        m_device.destroyAccelerationStructureKHR(tlas.acceleration_structure, nullptr, m_dyn_loader);
        destroy_buffer(tlas.buffer);
    }


    void device::destroy_acceleration_structure(const vk::AccelerationStructureKHR &accel) {

        m_device.destroyAccelerationStructureKHR(accel, nullptr, m_dyn_loader);
    }


    vk::AccelerationStructureGeometryDataKHR convert_to_vulkan_geometry(const geometry_data& geom) {

        vk::AccelerationStructureGeometryDataKHR out_geom = {};
        switch (geom.type) {

            case vk::GeometryTypeKHR::eTriangles: {

                return out_geom.setTriangles(vk::AccelerationStructureGeometryTrianglesDataKHR()
                                                .setVertexFormat(geom.vertex_format)
                                                .setVertexData(geom.data_addresses.vertex_dev_address)
                                                .setVertexStride(geom.stride)
                                                .setMaxVertex(geom.primitive_count * 3) // 3 vertices per triangle
                                                .setIndexType(geom.index_format)
                                                .setIndexData(geom.data_addresses.index_dev_address)
                                                .setTransformData(geom.data_addresses.transform_dev_address));
            }

            case vk::GeometryTypeKHR::eAabbs: {

                return out_geom.setAabbs(vk::AccelerationStructureGeometryAabbsDataKHR()
                    .setData(geom.data_addresses.aabb_dev_address)
                    .setStride(geom.stride));
            }
            default: break;
        }
        return out_geom;
    }

} // namespace vr
