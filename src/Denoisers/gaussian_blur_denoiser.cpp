
#include "../pch.h"

#include "vk_ray/denoisers/gaussian_blur_denoiser.h"

#include "gaussian_blur_denoiser.spv.h"


namespace vr
{
    namespace Denoise
    {
        GaussianBlurDenoiser::GaussianBlurDenoiser(vr::device *device, const DenoiserSettings &settings)
            : DenoiserInterface(device, settings)
        {
            mDenoiserParams = new Parameters();

            Init();
        }

        GaussianBlurDenoiser::~GaussianBlurDenoiser()
        {
            // Destroy descriptor set layout
            m_device->GetDevice().destroyDescriptorSetLayout(mDescriptorSetLayout);
            m_device->DestroyBuffer(mDescriptorBuffer.Buffer);

            // Destroy pipeline
            m_device->GetDevice().destroyPipeline(mPipeline);
            m_device->GetDevice().destroyPipelineLayout(mPipelineLayout);

            ->GetDevice().destroyShaderModule(mShaderModule);

            delete (Parameters *)mDenoiserParams;
        }

        void GaussianBlurDenoiser::Init()
        {
            // Create the resources
            auto resources = GetRequiredResources();
            DenoiserInterface::CreateResources(resources, mSettings.InputUsage, mSettings.OutputUsage);

            mdescriptor_items = {
                // Input image
                vr::descriptor_item(0, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eCompute, 1,
                                   &mInputResources[0].AccessImage),
                // Output image
                vr::descriptor_item(1, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eCompute, 1,
                                   &mOutputResources[0].AccessImage)};

            mDescriptorSetLayout = m_device->CreateDescriptorSetLayout(mdescriptor_items);
            mDescriptorBuffer = m_device->CreateDescriptorBuffer(mDescriptorSetLayout, mdescriptor_items,
                                                                vr::DescriptorBufferType::Combined);

            m_device->UpdateDescriptorBuffer(mDescriptorBuffer, mdescriptor_items, vr::DescriptorBufferType::Combined);

            // Create the pipeline layout
            auto pushConstantRange = vk::PushConstantRange()
                                         .setStageFlags(vk::ShaderStageFlagBits::eCompute)
                                         .setOffset(0)
                                         .setSize(sizeof(PushConstantData));
            vk::PipelineLayoutCreateInfo pipelineLayoutInfo = vk::PipelineLayoutCreateInfo()
                                                                  .setSetLayoutCount(1)
                                                                  .setPSetLayouts(&mDescriptorSetLayout)
                                                                  .setPPushConstantRanges(&pushConstantRange)
                                                                  .setPushConstantRangeCount(1);

            mPipelineLayout = m_device->GetDevice().createPipelineLayout(pipelineLayoutInfo);

            // From char array to uint32_t array
            uint32_t *shaderCode = (uint32_t *)&g_GaussianBlurDenoiser_main;

            // Spirv always has a size that is a multiple of 4
            uint32_t spvSize = sizeof(g_GaussianBlurDenoiser_main);

            // Create the shader module
            auto shaderModuleInfo = vk::ShaderModuleCreateInfo().setCodeSize(spvSize).setPCode(shaderCode);
            mShaderModule = m_device->GetDevice().createShaderModule(shaderModuleInfo);

            //  Create the pipeline
            auto pipelineInfo = vk::ComputePipelineCreateInfo()
                                    .setFlags(vk::PipelineCreateFlagBits::eDescriptorBufferEXT)
                                    .setLayout(mPipelineLayout)
                                    .setStage(vk::PipelineShaderStageCreateInfo()
                                                  .setStage(vk::ShaderStageFlagBits::eCompute)
                                                  .setModule(mShaderModule)
                                                  .setPName("GaussianBlurDenoiser_main"));

            auto res = m_device->GetDevice().createComputePipeline(nullptr, pipelineInfo);

            if (res.result != vk::Result::eSuccess)
                VR_LOG(error, "Failed to create median denoiser pipeline");
            mPipeline = res.value;
        }

        std::vector<Resource> GaussianBlurDenoiser::GetRequiredResources()
        {
            std::vector<Resource> resources(2);
            resources[0].Type = ResourceType::InputGeneral; // Median denoiser can be used in any context
            resources[0].Format = vk::Format::eR32G32B32A32Sfloat;
            resources[0].Usage = vk::ImageUsageFlagBits::eSampled;       // Want to sample the input image
            resources[0].AccessImage.Layout = vk::ImageLayout::eGeneral; // General layout for the output image

            resources[1].Type = ResourceType::OutputFinal;
            resources[1].Format = vk::Format::eR32G32B32A32Sfloat;
            resources[1].Usage = vk::ImageUsageFlagBits::eStorage;       // Storage image for the output
            resources[1].AccessImage.Layout = vk::ImageLayout::eGeneral; // General layout for the output image
            return resources;
        }

        void GaussianBlurDenoiser::Denoise(vk::CommandBuffer cmdBuffer)
        {
            // Bind the pipeline
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, mPipeline);
            m_device->BindDescriptorBuffer({mDescriptorBuffer}, cmdBuffer);
            m_device->BindDescriptorSet(mPipelineLayout, 0, 0, 0, cmdBuffer, vk::PipelineBindPoint::eCompute);

            PushConstantData pushData;
            pushData.Params.Radius = ((Parameters *)mDenoiserParams)->Radius;
            pushData.Params.Sigma = ((Parameters *)mDenoiserParams)->Sigma;
            pushData.Width = mSettings.Width;
            pushData.Height = mSettings.Height;

            // Push constants
            cmdBuffer.pushConstants(mPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstantData),
                                    &pushData);

            cmdBuffer.dispatch(mSettings.Width / 16, mSettings.Height / 16, 1);
        }
    } // namespace Denoise
} // namespace vr
