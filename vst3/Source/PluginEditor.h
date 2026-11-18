#pragma once
#include "PluginProcessor.h"

class AIMusicEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit AIMusicEditor(AIMusicProcessor&);
    ~AIMusicEditor() override { stopTimer(); }
    void resized() override;

private:
    void timerCallback() override;
    void browse(juce::TextEditor& target, const juce::String& pattern);

    AIMusicProcessor& proc;
    juce::Label lEngine { {}, "Engine (AIMusicGenerator-cli.exe)" }, lVocal { {}, "Vocal file (wav/flac/mp3)" },
        lSf { {}, "SoundFont (.sf2, 'auto' or 'none')" }, lGenre { {}, "Genre" }, lBars { {}, "Bars" }, status;
    juce::TextEditor engine, vocal, sf, genre, bars;
    juce::TextButton bEngine { "..." }, bVocal { "..." }, bSf { "..." }, go { "Generate" };
    std::unique_ptr<juce::FileChooser> chooser;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AIMusicEditor)
};
