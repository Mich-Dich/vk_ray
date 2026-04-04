
#include "../pch.h"

#include "vk_ray/denoisers/denoiser_interface.h"


namespace vr::Denoise
{

    denoiser_interface::~denoiser_interface() {

        auto vulkan_device = m_device->get_device();

        // lambda to destroy image views and samplers
        auto destroyImg = [this, vulkan_device](Resource &resource) {

            vulkan_device.destroyImageView(resource.access_image.View);
            vulkan_device.destroySampler(resource.access_image.Sampler);
            m_device->DestroyImage(resource.alloc_image);
        };

        for (auto &resource : m_input_resources)      destroyImg(resource);
        for (auto &resource : m_output_resources)     destroyImg(resource);
        for (auto &resource : m_internal_resources)   destroyImg(resource);
    }


    void denoiser_interface::create_resources(std::vector<Resource>& resources, vk::ImageUsageFlags input_usage, vk::ImageUsageFlags output_usage) {

        auto vulkan_device = m_device->GetDevice();
        auto max_anisotropy = m_device->GetProperties().limits.maxSamplerAnisotropy;

        // ALl resources are images
        auto image_info = vk::ImageCreateInfo()
            .setImageType(vk::ImageType::e2D)
            .setFormat(vk::Format::eUndefined) // Format depends on the resource type
            .setExtent(vk::Extent3D(m_settings.Width, m_settings.Height, 1))
            .setMipLevels(1)
            .setArrayLayers(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setTiling(vk::ImageTiling::eOptimal)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setInitialLayout(vk::ImageLayout::eUndefined);

        for (auto &resource : resources) {

            image_info.format = resource.format;
            image_info.usage = resource.usage | ((int)resource.type & 0b01 ? input_usage : output_usage); // set the usage depending on the type of resource

            resource.alloc_image = m_device->create_image(image_info, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

            // Create Image View
            auto viewInfo =
                vk::ImageViewCreateInfo()
                    .setImage(resource.AllocImage.Image)
                    .setViewType(vk::ImageViewType::e2D)
                    .setFormat(image_info.format)
                    .setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

            resource.access_image.View = vulkan_device.createImageView(viewInfo);

            // Only create sampler if the resource is sampled
            if (resource.usage & vk::ImageUsageFlagBits::eSampled)
            {
                // Create Sampler
                auto sampler_info = vk::SamplerCreateInfo()
                    .setMagFilter(vk::Filter::eLinear)
                    .setMinFilter(vk::Filter::eLinear)
                    .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
                    .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
                    .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
                    .setAnisotropyEnable(true)
                    .setMaxAnisotropy(max_anisotropy) // No performance hit
                    .setBorderColor(vk::BorderColor::eFloatOpaqueWhite)
                    .setUnnormalizedCoordinates(false)
                    .setCompareEnable(false)
                    .setCompareOp(vk::CompareOp::eNever)
                    .setMipmapMode(vk::SamplerMipmapMode::eLinear)
                    .setMipLodBias(0.0f)
                    .setMinLod(0.0f)
                    .setMaxLod(1.0f);

                resource.access_image.Sampler = vulkan_device.createSampler(samplerInfo);
            }

            // Add the resource to the correct vector
            if ((int)resource.type & 0b01) // if first bit is set, so it's an input
                m_input_resources.push_back(resource);

            else if ((int)resource.type & 0b10) // if second bit is set, so it's an output
                m_output_resources.push_back(resource);

            else
                m_internal_resources.push_back(resource);
        }
    }

} // namespace vr::Denoise