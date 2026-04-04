#pragma once

#include "vk_ray/shader.h"
#include "vk_ray/buffer.h"

#include "../../src/pch.h"

// FORWARD DECLARATIONS ================================================================================================

namespace vr {

    // CONSTANTS =======================================================================================================

    // MACROS ==========================================================================================================

    // TYPES ===========================================================================================================

    // brief Enum that defines the type of shader in the shader binding table
    enum class shader_group : uint8_t
    {
        ray_gen = 0,
        miss,
        hit_group,
        callable
    };


    // brief Contains all shaders that will be used in a hit group of an SBT
    struct hit_group
    {
        shader                                  closest_hit_shader = {};
        shader                                  any_hit_shader = {};
        shader                                  intersection_shader = {};
    };


    // brief Structure that defines a shader Binding Table that can be used to trace rays
    struct sbt_buffer
    {
        allocated_buffer                        ray_gen_buffer = {};
        allocated_buffer                        miss_buffer = {};
        allocated_buffer                        hit_group_buffer = {};
        allocated_buffer                        callable_buffer = {};

        /*
        Offsets define the start of the shader records in bytes.
        Next shader record is at offset + ShaderBindingTable::{ShaderType}GroupSize
        */

        vk::StridedDeviceAddressRegionKHR       ray_gen_region = {};
        vk::StridedDeviceAddressRegionKHR       miss_region = {};
        vk::StridedDeviceAddressRegionKHR       hit_group_region = {};
        vk::StridedDeviceAddressRegionKHR       callable_region = {};
    };


    // brief Pipeline library input structure
    struct ray_tracing_shader_collection
    {
        std::vector<shader>                     ray_gen_shaders = {};
        std::vector<shader>                     miss_shaders = {};
        std::vector<hit_group>                  hit_groups = {};
        std::vector<shader>                     callable_shaders = {};

        // brief Pipeline library that contains all the shaders in the collection
        // Filled by vk_ray when creating the pipeline library.
        // note Doesn't need to be destroyed, because it is destroyed when the pipeline it is linked to is destroyed
        vk::Pipeline collection_pipeline = nullptr;

        /*
        The order of how the shaders in the pipeline are laid out in the pipeline library.
        If there are no shaders of a certain type, the next shader type will be laid out where it would have been.
        So if no miss shaders are present, the hit groups will be after the ray gen shaders.
        | ray_gen_shaders   |
        | miss_shaders      |
        | hit_groups        |
        | callable_shaders  |
        */
    };


    // brief Structure that defines the settings for a ray tracing pipeline. When creating pipeline libraries that
    //  link to a single pipeline, the settigns should be the same for all pipelines
    struct pipeline_settings
    {
        vk::PipelineLayout                      pipeline_layout = nullptr;

        uint32_t                                max_recursion_depth = 1;            // brief The maximum number of levels of recursion allowed in the pipeline
        uint32_t                                max_payload_size = 0;               // brief The maximum size of the payload in bytes
        uint32_t                                max_hit_attribute_size = 0;         // brief The maximum size of the hit attribute in bytes in HitGroups
    };


    // brief Structure that defines the information needed to create a shader binding table
    struct sbt_info
    {
        uint32_t                                ray_gen_shader_record_size = 0;     // brief The size of each ray gen shader record in bytes
        uint32_t                                miss_shader_record_size = 0;        // brief The size of each miss shader record in bytes
        uint32_t                                hit_group_record_size = 0;          // brief The size of each hit group shader record in bytes
        uint32_t                                callable_shader_record_size = 0;    // brief The size of each callable shader record in bytes

        // If expecting more shaders than the pipeline has, reserve space for them
        // The SBT for each buffer will contain space for (***Indices.size() + Reserve***)

        uint32_t                                reserve_ray_gen_groups = 0;
        uint32_t                                reserve_miss_groups = 0;
        uint32_t                                reserve_hit_groups = 0;
        uint32_t                                reserve_callable_groups = 0;

        /*
        Graph of how the shaders might be mixed in a full pipeline.
        The shaders can be mixed in any way, but this is just an example

        | Pipeline                       | SBT Index |
        ---------------------------------------------
        | shader_collection1.rgen        | 0         |
        | shader_collection1.miss        | 1         |
        | shader_collection1.hit_group   | 2         |
        | shader_collection1.callable    | 3         |
        | shader_collection2.miss        | 4         |
        | shader_collection2.hit_group1  | 5         |
        | shader_collection2.hit_group2  | 6         |
        | shader_collection2.callable    | 7         |
        | shader_collection3.rgen        | 8         |
        ---------------------------------------------
        We want to support linking new shader collections to the already existing pipeline for efficiency.
        This means that the shaders can be mixed in any way and organising them is necessary for the SBT to work.
        We need indices for shader types to tell where in the pipeline the shaders live.
        Then the shaders go into the SBT organised by their index.
        That would turn the graph into this.
        | Pipeline                       | SBT Index |
        ---------------------------------------------
        | shader_collection1.rgen        | 0         |
        | shader_collection3.rgen        | 1         |
        | shader_collection1.miss        | 2         |
        | shader_collection2.miss        | 3         |
        | shader_collection1.hit_group   | 4         |
        | shader_collection2.hit_group1  | 5         |
        | shader_collection2.hit_group2  | 6         |
        | shader_collection1.callable    | 7         |
        | shader_collection2.callable    | 8         |
        ---------------------------------------------
        */

        std::vector<uint32_t>                   ray_gen_indices = {};       // brief These vectors contain where the raygen shaders live in the pipeline
        std::vector<uint32_t>                   miss_indices = {};          // brief These vectors where the miss shaders live in the pipeline
        std::vector<uint32_t>                   hit_group_indices = {};     // brief These vectors where the hit group shaders live in the pipeline
        std::vector<uint32_t>                   callable_indices = {};      // brief These vectors where the callable shaders live in the pipeline
    };

    // STATIC VARIABLES ================================================================================================

    // FUNCTION DECLARATION ============================================================================================

    // TEMPLATE DECLARATION ============================================================================================

    // CLASS DECLARATION ===============================================================================================

}
