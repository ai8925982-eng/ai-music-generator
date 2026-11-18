#include "PluginEditor.h"

AIMusicEditor::AIMusicEditor(AIMusicProcessor& p) : AudioProcessorEditor(&p), proc(p)
{
    setSize(560, 330);
    for (auto* c : std::initializer_list<juce::Component*>{ &lEngine, &lVocal, &lSf, &lGenre, &lBars, &status,
                                                            &engine, &vocal, &sf, &genre, &bars, &bEngine, &bVocal, &bSf, &go })
        addAndMakeVisible(c);
    engine.setText(p.enginePath); vocal.setText(p.vocalPath); sf.setText(p.soundfontPath);
    genre.setText(p.genre); bars.setText(juce::String(p.bars));
    status.setJustificationType(juce::Justification::topLeft);
    bEngine.onClick = [this] { browse(engine, "*.exe"); };
    bVocal.onClick = [this] { browse(vocal, "*.wav;*.flac;*.mp3;*.ogg"); };
    bSf.onClick = [this] { browse(sf, "*.sf2"); };
    go.onClick = [this]
    {
        proc.enginePath = engine.getText(); proc.vocalPath = vocal.getText(); proc.soundfontPath = sf.getText();
        proc.genre = genre.getText().trim().isEmpty() ? "isan" : genre.getText().trim();
        proc.bars = juce::jlimit(8, 128, bars.getText().getIntValue());
        proc.startGenerate();
    };
    startTimerHz(4);
}

void AIMusicEditor::browse(juce::TextEditor& target, const juce::String& pattern)
{
    chooser = std::make_unique<juce::FileChooser>("Choose file", juce::File(), pattern);
    chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                         [this, &target](const juce::FileChooser& fc)
                         { if (fc.getResult().existsAsFile()) target.setText(fc.getResult().getFullPathName()); });
}

void AIMusicEditor::timerCallback()
{
    status.setText(proc.status(), juce::dontSendNotification);
    go.setEnabled(!proc.busy());
}

void AIMusicEditor::resized()
{
    auto r = getLocalBounds().reduced(12);
    auto row = [&](juce::Label& l, juce::TextEditor& e, juce::TextButton* b)
    {
        l.setBounds(r.removeFromTop(20));
        auto line = r.removeFromTop(26);
        if (b) b->setBounds(line.removeFromRight(32));
        e.setBounds(line);
        r.removeFromTop(6);
    };
    row(lEngine, engine, &bEngine);
    row(lVocal, vocal, &bVocal);
    row(lSf, sf, &bSf);
    auto half = r.removeFromTop(20);
    lGenre.setBounds(half.removeFromLeft(getWidth() / 2 - 12)); lBars.setBounds(half);
    auto line = r.removeFromTop(26);
    genre.setBounds(line.removeFromLeft(getWidth() / 2 - 18)); line.removeFromLeft(6); bars.setBounds(line);
    r.removeFromTop(8);
    go.setBounds(r.removeFromTop(32));
    r.removeFromTop(6);
    status.setBounds(r);
}
