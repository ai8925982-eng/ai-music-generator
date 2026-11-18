cmake_minimum_required(VERSION 3.22)
project(AIMusicGeneratorPlugin VERSION 1.0.0)

include(FetchContent)
FetchContent_Declare(JUCE GIT_REPOSITORY https://github.com/juce-framework/JUCE.git GIT_TAG 8.0.4 GIT_SHALLOW ON)
FetchContent_MakeAvailable(JUCE)

juce_add_plugin(AIMusicGeneratorPlugin
    PRODUCT_NAME "AI Music Generator"
    COMPANY_NAME "AIMusicGenerator"
    PLUGIN_MANUFACTURER_CODE Aimg
    PLUGIN_CODE Aim1
    FORMATS VST3 Standalone
    IS_SYNTH TRUE
    NEEDS_MIDI_OUTPUT TRUE
    NEEDS_MIDI_INPUT FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE)

target_sources(AIMusicGeneratorPlugin PRIVATE Source/PluginProcessor.cpp Source/PluginEditor.cpp)
target_compile_features(AIMusicGeneratorPlugin PRIVATE cxx_std_17)
target_compile_definitions(AIMusicGeneratorPlugin PUBLIC JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)
target_link_libraries(AIMusicGeneratorPlugin PRIVATE
    juce::juce_audio_utils juce::juce_audio_processors
    PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_lto_flags juce::juce_recommended_warning_flags)
