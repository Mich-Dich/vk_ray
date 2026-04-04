#pragma once

#include "vk_ray/denoisers/denoiser_interface.h"

// FORWARD DECLARATIONS ================================================================================================

namespace vr::denoise {

    // CONSTANTS =======================================================================================================

    // MACROS ==========================================================================================================

    // TYPES ===========================================================================================================

    // STATIC VARIABLES ================================================================================================

    // FUNCTION DECLARATION ============================================================================================

    // TEMPLATE DECLARATION ============================================================================================

    // CLASS DECLARATION ===============================================================================================

    class gaussian_blur_denoiser : public denoiser_interface {
    public:

        gaussian_blur_denoiser(vr::device *device, const denoiser_settings &settings);
        ~gaussian_blur_denoiser() override;

        gaussian_blur_denoiser() = delete;
        gaussian_blur_denoiser(const gaussian_blur_denoiser &) = delete;

        std::vector<Resource> get_required_resources() override;

        void denoise(vk::CommandBuffer cmdBuffer) override;

        // The settings for the gaussian blur
        struct parameters {

            uint32_t    radius = 3;       // The radius of the gaussian blur
            float       sigma = 1.0f;     // The sigma value for the gaussian blur, smoothness
        };

    private:

        // Data for the push constants
        struct push_constant_data {

            uint32_t    width;
            uint32_t    height;
            parameters  params;
        };

        void init();

        std::vector<descriptor_item>            m_descriptor_items = {};
        vk::DescriptorSetLayout                 m_descriptor_set_layout = nullptr;
        descriptor_buffer                       m_descriptor_buffer = {};

        vk::Pipeline                            m_pipeline = nullptr;
        vk::PipelineLayout                      m_pipeline_layout = nullptr;

        vk::ShaderModule                        m_shader_module = nullptr;

    };

}
