#pragma once

#include "vk_ray/buffer.h"

// FORWARD DECLARATIONS ================================================================================================

namespace vr { class device; }


namespace vr::denoise {

    // CONSTANTS =======================================================================================================

    // MACROS ==========================================================================================================

    // TYPES ===========================================================================================================

    enum class resource_type : uint32_t
    {
        input = 0b01,
        output = 0b10,
        internal = 0b11,
        input_general = input | 0b100,      // Input resource that can be anything
        output_final = output | 0b100,      // Output resource that is the final image
    };


    struct resource
    {
        resource() = default;

        allocated_image         alloc_image;
        accessible_image        access_image;

        vk::ImageUsageFlags     usage;
        resource_type           type;
        vk::Format              format;
    };


    struct denoiser_settings
    {
        uint32_t                width;
        uint32_t                height;

        vk::ImageUsageFlags     input_usage;
        vk::ImageUsageFlags     output_usage;
    };

    // STATIC VARIABLES ================================================================================================

    // FUNCTION DECLARATION ============================================================================================

    // TEMPLATE DECLARATION ============================================================================================

    // CLASS DECLARATION ===============================================================================================

    class denoiser_interface {
    public:

        denoiser_interface(vr::device *device, const DenoiserSettings &settings)
            : m_device(device), m_settings(settings) { }

        virtual ~denoiser_interface();

        // Delete unwanted constructors
        denoiser_interface() = delete;
        denoiser_interface(const denoiser_interface &) = delete;

        // TODO: Let the user allocate the resources and pass them to the denoiser

        // Get the resources that are required for the denoiser
        // return The required resources
        // note The resources vector doesn't contain allocated images, only the required information to allocate
        // them
        virtual std::vector<Resource> get_required_resources() { return std::vector<Resource>(); }


        // Get the input resources that are used by the denoiser
        // return The input resources
        virtual const std::vector<Resource> &get_input_resources() const { return m_input_resources; }


        // Get the output resources that are used by the denoiser
        // return The internal resources
        virtual const std::vector<Resource> &get_output_resources() const { return m_output_resources; }


        // Denoise the image
        // param cmdBuffer The command buffer to use for the denoising, must be in recording state and this
        // function uses push constants for the settings
        virtual void denoise(vk::CommandBuffer cmdBuffer) {};


        // Get the Parameters struct
        // tparam T The type of the settings struct -> DenoiserX::Parameters
        // return The Parameters struct
        // warning If a settings struct from another denoiser is requested, it can cause in segmentation faults,
        // because the memory layout is different
        template <typename T>
        T get_denoiser_params() const { return *(T*)m_denoiser_params; }


        // Set the Parameters struct
        // tparam T The type of the Parameters struct -> DenoiserX::Parameters
        // param params The Parameters struct
        // warning If a Parameters struct from another denoiser is set, it can cause in segmentation faults,
        // because the memory layout is different
        template <typename T>
        void set_denoiser_params(const T &params) { *(T*)m_denoiser_params = params; }

    protected:

        void CreateResources(std::vector<Resource>& resources, vk::ImageUsageFlags input_usage, vk::ImageUsageFlags output_usage);

        vr::device*                     m_device;                       // Reference to the vulray device that was used to create the denoiser
        denoiser_settings               m_settings;
        std::vector<resource>           m_input_resources;
        std::vector<resource>           m_internal_resources;
        std::vector<resource>           m_output_resources;
        void*                           m_denoiser_params = nullptr;    // pointer to the settings struct, allocated by a derived class

    };

    // Type alias for the denoiser interface
    using denoiser = std::unique_ptr<Denoise::denoiser_interface>;

}
