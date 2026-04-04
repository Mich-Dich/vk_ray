
#include "pch.h"

#include "vk_ray/shader.h"
#include "vk_ray/device.h"

// FORWARD DECLARATIONS ================================================================================================

namespace vr {

    // CONSTANTS =======================================================================================================

    // MACROS ==========================================================================================================

    // TYPES ===========================================================================================================

    // STATIC VARIABLES ================================================================================================

    // FUNCTION IMPLEMENTATION =========================================================================================

    // CLASS IMPLEMENTATION ============================================================================================

    // CLASS PUBLIC ====================================================================================================

    std::pair<std::vector<vk::PipelineShaderStageCreateInfo>, std::vector<vk::RayTracingShaderGroupCreateInfoKHR>>
    device::get_shader_stages_and_ray_tracing_groups(const ray_tracing_shader_collection &info)
    {
        std::vector<vk::PipelineShaderStageCreateInfo> shader_stages;
        std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shader_groups;
        shader_stages.reserve(1 + info.miss_shaders.size() + info.hit_groups.size() + info.callable_shaders.size());
        shader_groups.reserve(1 + info.miss_shaders.size() + info.hit_groups.size() + info.callable_shaders.size());

        // create ray gen shader groups
        for (auto &shader : info.ray_gen_shaders) {

            shader_stages.push_back(vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eRaygenKHR)
                .setModule(shader.module)
                .setPName(shader.entry_point));

            uint32_t rayGenIndex = static_cast<uint32_t>(shader_stages.size() - 1);

            shader_groups.push_back(vk::RayTracingShaderGroupCreateInfoKHR()
                .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
                .setGeneralShader(rayGenIndex)
                .setClosestHitShader(VK_SHADER_UNUSED_KHR)
                .setAnyHitShader(VK_SHADER_UNUSED_KHR)
                .setIntersectionShader(VK_SHADER_UNUSED_KHR));
        }

        // create miss shader groups
        for (auto &shader : info.miss_shaders) {

            shader_stages.push_back(vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eMissKHR)
                .setModule(shader.module)
                .setPName(shader.entry_point));

            uint32_t missIndex = static_cast<uint32_t>(shader_stages.size() - 1);

            shader_groups.push_back(vk::RayTracingShaderGroupCreateInfoKHR()
                .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
                .setGeneralShader(missIndex)
                .setClosestHitShader(VK_SHADER_UNUSED_KHR)
                .setAnyHitShader(VK_SHADER_UNUSED_KHR)
                .setIntersectionShader(VK_SHADER_UNUSED_KHR));
        }

        // create hit group shader groups
        for (auto& group : info.hit_groups) {

            // null init hit group, will be filled in later
            auto hitGroup = vk::RayTracingShaderGroupCreateInfoKHR()
                // assume triangles for now, change if hit group has a custom intersection shader
                .setType(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup)
                .setGeneralShader(VK_SHADER_UNUSED_KHR)
                .setClosestHitShader(VK_SHADER_UNUSED_KHR)
                .setAnyHitShader(VK_SHADER_UNUSED_KHR)
                .setIntersectionShader(VK_SHADER_UNUSED_KHR);

            // check if both shaders are null
            if (!group.closest_hit_shader.module && !group.any_hit_shader.module && !group.intersection_shader.module) {

                VR_LOG(error, "CreateRayTracingPipeline: Hit group must have at least one shader");
            }

            // add closest hit shader if it exists
            if (group.closest_hit_shader.module) {

                shader_stages.push_back(vk::PipelineShaderStageCreateInfo()
                    .setStage(vk::ShaderStageFlagBits::eClosestHitKHR)
                    .setModule(group.closest_hit_shader.module)
                    .setPName(group.closest_hit_shader.entry_point));

                uint32_t closestHitIndex = static_cast<uint32_t>(shader_stages.size() - 1);
                hitGroup.setClosestHitShader(closestHitIndex);
            }

            // add any hit shader if it exists
            if (group.any_hit_shader.module) {

                shader_stages.push_back(vk::PipelineShaderStageCreateInfo()
                    .setStage(vk::ShaderStageFlagBits::eAnyHitKHR)
                    .setModule(group.any_hit_shader.module)
                    .setPName(group.any_hit_shader.entry_point));

                uint32_t anyHitIndex = static_cast<uint32_t>(shader_stages.size() - 1);
                hitGroup.setAnyHitShader(anyHitIndex);
            }

            // add intersection shader if it exists
            if (group.intersection_shader.module) {

                shader_stages.push_back(vk::PipelineShaderStageCreateInfo()
                    .setStage(vk::ShaderStageFlagBits::eIntersectionKHR)
                    .setModule(group.intersection_shader.module)
                    .setPName(group.intersection_shader.entry_point));

                uint32_t intersectionIndex = static_cast<uint32_t>(shader_stages.size() - 1);
                hitGroup.setType(vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup);
                hitGroup.setIntersectionShader(intersectionIndex);
            }

            shader_groups.push_back(hitGroup);
        }
        // create callable shader groups
        for (auto &shader : info.callable_shaders) {

            shader_stages.push_back(vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eCallableKHR)
                .setModule(shader.module)
                .setPName(shader.entry_point));

            uint32_t callIndex = static_cast<uint32_t>(shader_stages.size() - 1);

            shader_groups.push_back(vk::RayTracingShaderGroupCreateInfoKHR()
                .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
                .setGeneralShader(callIndex)
                .setClosestHitShader(VK_SHADER_UNUSED_KHR)
                .setAnyHitShader(VK_SHADER_UNUSED_KHR)
                .setIntersectionShader(VK_SHADER_UNUSED_KHR));
        }
        return std::make_pair(std::move(shader_stages), std::move(shader_groups));
    }


    std::pair<vk::Pipeline, sbt_info> device::create_ray_tracing_pipeline(const ray_tracing_shader_collection& shader_collection, pipeline_settings& settings, vk::PipelineCreateFlags flags,
        vk::DeferredOperationKHR deferred_operation) {

        vr::sbt_info sbt_info = {};

        uint32_t pipelineIndex = 0; // index of shader in the compiled pipeline

        for ([[maybe_unused]] auto &shader : shader_collection.ray_gen_shaders)
            sbt_info.ray_gen_indices.push_back(pipelineIndex++);

        for ([[maybe_unused]] auto &shader : shader_collection.miss_shaders)
            sbt_info.miss_indices.push_back(pipelineIndex++);

        for ([[maybe_unused]] auto &shader : shader_collection.hit_groups)
            sbt_info.hit_group_indices.push_back(pipelineIndex++);

        for ([[maybe_unused]] auto &shader : shader_collection.callable_shaders)
            sbt_info.callable_indices.push_back(pipelineIndex++);

        auto [shader_stages, shder_groups] = get_shader_stages_and_ray_tracing_groups(shader_collection);

        vk::RayTracingPipelineInterfaceCreateInfoKHR interfaceInfo = vk::RayTracingPipelineInterfaceCreateInfoKHR()
                .setMaxPipelineRayHitAttributeSize(settings.max_hit_attribute_size)
                .setMaxPipelineRayPayloadSize(settings.max_payload_size);

        auto pipeline_info = vk::RayTracingPipelineCreateInfoKHR()
            .setFlags(flags)
            .setMaxPipelineRayRecursionDepth(settings.max_recursion_depth)
            .setPLibraryInterface(&interfaceInfo)
            .setLayout(settings.pipeline_layout)
            .setGroups(shder_groups)
            .setStages(shader_stages);

        auto res = m_device.createRayTracingPipelineKHR(deferred_operation, nullptr, pipeline_info, nullptr, m_dyn_loader);

        // when deferred_operation is not null, the pipeline is created asynchronously, so it doesn't return success or failure
        if (res.result != vk::Result::eSuccess && res.result != vk::Result::eOperationDeferredKHR) {

            VR_LOG(error, "CreateRayTracingPipeline: Failed to create ray tracing pipeline");
            res.value = nullptr;
        }

        return std::make_pair(res.value, sbt_info);
    }


    std::pair<vk::Pipeline, sbt_info> device::create_ray_tracing_pipeline(const std::vector<ray_tracing_shader_collection>& shader_collections, pipeline_settings& settings,
        vk::PipelineCreateFlags flags, vk::PipelineCache cache, vk::DeferredOperationKHR deferred_operation) {

        vr::sbt_info sbt_info = {};
        std::vector<vk::Pipeline> lib_pipelines = {};
        lib_pipelines.reserve(shader_collections.size());
        uint32_t pipelineIndex = 0; // index of shader in the compiled pipeline
        for (auto &collection : shader_collections)
        {
            lib_pipelines.push_back(collection.collection_pipeline);
            // Assign where in the pipeline the shaders are, for future SBT creation,
            // so opaque handles can be queried for the shader groups
            for ([[maybe_unused]] auto &shader : collection.ray_gen_shaders)
                sbt_info.ray_gen_indices.push_back(pipelineIndex++);

            for ([[maybe_unused]] auto &shader : collection.miss_shaders)
                sbt_info.miss_indices.push_back(pipelineIndex++);

            for ([[maybe_unused]] auto &shader : collection.hit_groups)
                sbt_info.hit_group_indices.push_back(pipelineIndex++);

            for ([[maybe_unused]] auto &shader : collection.callable_shaders)
                sbt_info.callable_indices.push_back(pipelineIndex++);
        }

        vk::RayTracingPipelineInterfaceCreateInfoKHR interfaceInfo = vk::RayTracingPipelineInterfaceCreateInfoKHR()
                .setMaxPipelineRayHitAttributeSize(settings.max_hit_attribute_size)
                .setMaxPipelineRayPayloadSize(settings.max_payload_size);

        vk::PipelineLibraryCreateInfoKHR libraryInfo = vk::PipelineLibraryCreateInfoKHR().setPLibraries(lib_pipelines.data()).setLibraryCount(lib_pipelines.size());

        auto pipeline_info = vk::RayTracingPipelineCreateInfoKHR()
            .setFlags(flags)
            .setMaxPipelineRayRecursionDepth(settings.max_recursion_depth)
            .setPLibraryInterface(&interfaceInfo)
            .setPLibraryInfo(&libraryInfo)
            .setLayout(settings.pipeline_layout);

        auto res = m_device.createRayTracingPipelineKHR(deferred_operation, cache, pipeline_info, nullptr, m_dyn_loader);
        // when deferred_operation is not null, the pipeline is created asynchronously, so it doesn't return success or failure
        if (res.result != vk::Result::eSuccess && res.result != vk::Result::eOperationDeferredKHR) {

            VR_LOG(error, "CreateRayTracingPipeline: Failed to create ray tracing pipeline");
            res.value = nullptr;
        }

        return std::make_pair(res.value, sbt_info);
    }


    std::pair<vk::Pipeline, sbt_info> device::create_ray_tracing_pipeline(const std::vector<ray_tracing_shader_collection>& shader_collections, pipeline_settings& settings,
        sbt_info& sbtInfoOld, vk::PipelineCreateFlags flags, vk::PipelineCache cache, vk::DeferredOperationKHR deferred_operation) {

        std::pair<vk::Pipeline, sbt_info> pipeline_info = create_ray_tracing_pipeline(shader_collections, settings, flags, cache, deferred_operation);
        pipeline_info.second.ray_gen_shader_record_size = sbtInfoOld.ray_gen_shader_record_size;
        pipeline_info.second.miss_shader_record_size = sbtInfoOld.miss_shader_record_size;
        pipeline_info.second.hit_group_record_size = sbtInfoOld.hit_group_record_size;
        pipeline_info.second.callable_shader_record_size = sbtInfoOld.callable_shader_record_size;

        return pipeline_info;
    }


    void device::create_pipeline_library(ray_tracing_shader_collection& shader_collection, pipeline_settings& settings, vk::PipelineCreateFlags flags, vk::PipelineCache cache,
        vk::DeferredOperationKHR deferred_operation) {

        vk::RayTracingPipelineInterfaceCreateInfoKHR interfaceInfo = vk::RayTracingPipelineInterfaceCreateInfoKHR()
            .setMaxPipelineRayHitAttributeSize(settings.max_hit_attribute_size)
            .setMaxPipelineRayPayloadSize(settings.max_payload_size);

        auto [shader_stages, shder_groups] = get_shader_stages_and_ray_tracing_groups(shader_collection);

        auto pipeline_info = vk::RayTracingPipelineCreateInfoKHR()
            .setFlags(flags | vk::PipelineCreateFlagBits::eLibraryKHR)
            .setMaxPipelineRayRecursionDepth(settings.max_recursion_depth)
            .setPLibraryInterface(&interfaceInfo)
            .setLayout(settings.pipeline_layout)
            .setGroups(shder_groups)
            .setStages(shader_stages);

        auto res = m_device.createRayTracingPipelineKHR(deferred_operation, cache, pipeline_info, nullptr, m_dyn_loader);

        // when deferred_operation is not null, the pipeline is created asynchronously, so it doesn't return success or failure
        if (res.result != vk::Result::eSuccess && res.result != vk::Result::eOperationDeferredKHR)
        {
            VR_LOG(error, "CreateRayTracingPipeline: Failed to create ray tracing pipeline");
            res.value = nullptr;
        }

        shader_collection.collection_pipeline = res.value;
    }


    void device::dispatch_rays(const vk::Pipeline rtPipeline, const sbt_buffer& buffer, uint32_t width, uint32_t height, uint32_t depth, vk::CommandBuffer command_buffer) {

        command_buffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, rtPipeline);
        command_buffer.traceRaysKHR(&buffer.ray_gen_region, &buffer.miss_region, &buffer.hit_group_region, &buffer.callable_region, width, height, depth, m_dyn_loader);
    }

    // CLASS PROTECTED =================================================================================================

    // CLASS PRIVATE ===================================================================================================

}
