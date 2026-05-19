
#include "pch.h"

#include "vk_ray/device.h"
#include "vk_ray/shader.h"

// FORWARD DECLARATIONS ================================================================================================

namespace vr {

    // CONSTANTS =======================================================================================================

    // MACROS ==========================================================================================================

    // TYPES ===========================================================================================================

    // STATIC VARIABLES ================================================================================================

    // FUNCTION IMPLEMENTATION =========================================================================================

    // CLASS IMPLEMENTATION ============================================================================================

    shader device::create_shader_from_spv(const std::vector<uint32_t>& spv) {

        shader outShader = {};
        if (spv.empty()) {

            LOG(error, "ShaderCreateInfo must have SPIRV code");
            return outShader; // return empty shader, because no shader was created
        }

        outShader.module = create_shader_module(spv);
        return outShader;
    }


    void device::destroy_shader(shader &shader) { m_device.destroyShaderModule(shader.module); }


    vk::ShaderModule device::create_shader_module(const std::vector<uint32_t> &spvCode) {

        // create shader module
        auto shaderModuleCreateInfo = vk::ShaderModuleCreateInfo().setCodeSize(spvCode.size() * sizeof(uint32_t)).setPCode(spvCode.data());
        auto shaderModule = m_device.createShaderModule(shaderModuleCreateInfo);
        return shaderModule;
    }

    // CLASS PUBLIC ====================================================================================================

    // CLASS PROTECTED =================================================================================================

    // CLASS PRIVATE ===================================================================================================

}
