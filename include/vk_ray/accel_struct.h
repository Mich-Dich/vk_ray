#pragma once

#include "vk_ray/buffer.h"

#include "../../src/pch.h"

// FORWARD DECLARATIONS ================================================================================================

namespace vr {

    // CONSTANTS =======================================================================================================

    // MACROS ==========================================================================================================

    // TYPES ===========================================================================================================

    // Contains the device address of the geometry data
    struct geometry_device_address
    {
        geometry_device_address() = default;
        geometry_device_address(vk::DeviceAddress vertex_or_aabb_dev_address, vk::DeviceAddress index_dev_address)
            : vertex_dev_address(vertex_or_aabb_dev_address), index_dev_address(index_dev_address) { }

        union {

            vk::DeviceAddress           vertex_dev_address;
            vk::DeviceAddress           aabb_dev_address;
        };

        vk::DeviceAddress               index_dev_address = {};       // Device address of the index buffer, only used for triangles
        vk::DeviceAddress               transform_dev_address = {};   // Buffer containing the transform for the geometry, if this is null, the geometry will use the identity matrix
    };

    struct geometry_data
    {
        vk::GeometryTypeKHR             type = vk::GeometryTypeKHR::eTriangles;         // Type of geometry, either triangles or AABBs
        geometry_device_address         data_addresses = {};                            // Buffer containing the vertices, only used for triangles
        vk::IndexType                   index_format = vk::IndexType::eUint32;          // Format of the index buffer, only used for triangles
        vk::Format                      vertex_format = vk::Format::eR32G32B32Sfloat;   // Format of the vertex buffer, only used for triangles
        uint32_t                        stride = 0;                                     // Stride of each element in the vertex buffer or AABB buffer
        uint32_t                        primitive_count = 0;                            // Number of primitives in the geometry, such as triangles or AABBs
        vk::GeometryFlagsKHR            flags = vk::GeometryFlagBitsKHR::eOpaque;       // Flags for the geometry, Default is eOpaque
    };

    //--------------------------------------------------------------------------------------
    // BLAS STRUCTURES
    //--------------------------------------------------------------------------------------

    struct blas_create_info
    {
        std::vector<geometry_data>                   geometries;     // Geometries to be added to the BLAS (All the geometries must be of the same type, either triangles or AABBs)

        // Flags for the acceleration structure, Default is ePreferFastTrace
        /// @note The flags must be appropriately set for future use, e.g., compaction, update, etc.
        vk::BuildAccelerationStructureFlagsKHR      flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
    };


    struct blas_build_info
    {
        vk::AccelerationStructureBuildSizesInfoKHR                      build_sizes = {};           // Contains the build sizes for the acceleration structure
        vk::AccelerationStructureBuildGeometryInfoKHR                   build_geometry_info = {};   // Contains the build info for the acceleration structure

        // The following are shared_ptr, to avoid std::vector reallocation when copied around
        // We could use std::vector, but it expects the function to be noexcept,
        // which reverts to copying, because the functions can throw exceptions

        std::shared_ptr<vk::AccelerationStructureGeometryKHR[]>         geometries = nullptr;       // Geometries that are included in the acceleration structure
        uint32_t                                                        geometry_count = 0;
        std::shared_ptr<vk::AccelerationStructureBuildRangeInfoKHR[]>   ranges = nullptr;           // Build ranges info for building the acceleration structure
        uint32_t                                                        ranges_count = 0;
    };


    struct blas_handle
    {
        vk::AccelerationStructureKHR                            acceleration_structure = nullptr;       // Raw handle of the acceleration structure
        allocated_buffer                                        buffer = {};                // Buffer containing the acceleration structure
    };


    struct blas_update_info
    {
        blas_handle*                                            source_blas = {};           // Indicates the destination BLAS which is getting updated
        blas_build_info                                         source_build_info = {};     // This is the build info that was given when creating the destination BLAS, which will be reused

        // If the updated geometries are in different buffers, NewGeometryAddresses will contain the device
        /// addresses of the primitives
        /// @note This field can be null if the updated geometries are in the same buffers as the source BLAS
        ///       The size of the vector must be the same as the number of geometries in @c SourceBuildInfo
        ///       Transform buffer must be provided if the source BLAS had a transform buffer, else it must be null
        std::vector<geometry_device_address>                    new_geometry_addresses = {};
    };


    struct compaction_request
    {
        vk::QueryPool                                           compaction_query_pool = nullptr; // Query pool that will be used to get the compacted size
        std::vector<vk::AccelerationStructureKHR>               source_blas = {};           // All the BLASes that will be compacted
    };

    //--------------------------------------------------------------------------------------
    // TLAS STRUCTUES
    //--------------------------------------------------------------------------------------

    struct tlas_create_info
    {
        uint32_t                                                max_instance_count = 0;     // Contains the geometries that will be added to the TLAS
        vk::DeviceAddress                                       instance_dev_address = {};  // Device address of the instance buffer
        vk::BuildAccelerationStructureFlagsKHR                  flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;        // Flags for the acceleration structure, Default is ePreferFastTrace
    };


    struct tlas_build_info
    {
        vk::AccelerationStructureBuildSizesInfoKHR              build_sizes = {};           // Contains the build sizes for the acceleration structure
        vk::AccelerationStructureBuildGeometryInfoKHR           build_geometry_info = {};   // Contains the build info for the acceleration structure
        std::shared_ptr<vk::AccelerationStructureGeometryKHR>   geometry = {};              // Geometries that are included in the acceleration structure
        vk::AccelerationStructureBuildRangeInfoKHR              range_info = {};            // Build ranges info for building the acceleration structure
        uint32_t                                                max_instance_count = 0;     // Max number of instances that can be added to the TLAS
    };


    struct tlas_handle
    {
        vk::AccelerationStructureKHR                            acceleration_structure = nullptr;   // Raw handle of the acceleration structure
        allocated_buffer                                        buffer = {};                // Buffer containing the acceleration structure
    };

    // STATIC VARIABLES ================================================================================================

    // FUNCTION DECLARATION ============================================================================================

    vk::AccelerationStructureGeometryDataKHR convert_to_vulkan_geometry(const geometry_data& geom);

    // TEMPLATE DECLARATION ============================================================================================

    // CLASS DECLARATION ===============================================================================================

}
