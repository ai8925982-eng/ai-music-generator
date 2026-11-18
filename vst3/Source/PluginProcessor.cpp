#include "PluginProcessor.h"
#include "PluginEditor.h"

AIMusicProcessor::AIMusicProcessor()
    : AudioProcessor(BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      juce::Thread("generate")
{
    workDir = juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("AIMusicGenerator_vst");
    workDir.createDirectory();
}
AIMusicProcessor::~AIMusicProcessor() { stopThread(60000); }

bool AIMusicProcessor::isBusesLayoutSupported(const BusesLayout& l) const
{
    return l.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
}

void AIMusicProcessor::prepareToPlay(double sr, int) { hostRate = sr; }

void AIMusicProcessor::startGenerate()
{
    if (!isThreadRunning()) startThread();
}

void AIMusicProcessor::run()
{
    juce::File exe(enginePath);
    if (!exe.existsAsFile()) { setStatus("Engine not found: set the path to AIMusicGenerator-cli.exe"); return; }
    setStatus("Generating... (can take a minute or two)");
    auto out = workDir.getChildFile("out");
    out.deleteRecursively();
    out.createDirectory();

    juce::StringArray args;
    args.add(exe.getFullPathName());
    args.add("--genre"); args.add(genre);
    args.add("--mode"); args.add("audio_midi");
    args.add("--bars"); args.add(juce::String(bars));
    args.add("--output"); args.add(out.getFullPathName());
    args.add("--name"); args.add("song");
    args.add("--soundfont"); args.add(soundfontPath.isEmpty() ? "none" : soundfontPath);
    if (vocalPath.isNotEmpty()) { args.add("--vocal"); args.add(vocalPath); }

    juce::ChildProcess p;
    if (!p.start(args)) { setStatus("Could not start the engine"); return; }
    while (p.isRunning() && !threadShouldExit()) wait(200);
    if (threadShouldExit()) { p.kill(); return; }
    auto log = p.readAllProcessOutput();
    auto wav = out.getChildFile("song.wav"), mid = out.getChildFile("song.mid");
    if (!wav.existsAsFile()) { setStatus("Engine finished without audio:\n" + log.trimEnd().getLastCharacters(300)); return; }
    loadResult(wav, mid);
    setStatus("Done. Press play in the host (the song follows the transport).");
}

void AIMusicProcessor::loadResult(const juce::File& wav, const juce::File& mid)
{
    juce::AudioFormatManager fm;
    fm.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> r(fm.createReaderFor(wav));
    if (!r) { setStatus("Could not read the generated wav"); return; }
    juce::AudioBuffer<float> b((int) juce::jmax(2u, r->numChannels), (int) r->lengthInSamples);
    r->read(&b, 0, (int) r->lengthInSamples, 0, true, true);
    if (r->numChannels == 1) b.copyFrom(1, 0, b, 0, 0, b.getNumSamples());
    juce::MidiMessageSequence seq;
    if (mid.existsAsFile())
    {
        juce::FileInputStream in(mid);
        juce::MidiFile mf;
        if (in.openedOk() && mf.readFrom(in))
        {
            mf.convertTimestampTicksToSeconds();
            for (int t = 0; t < mf.getNumTracks(); ++t) seq.addSequence(*mf.getTrack(t), 0.0);
            seq.updateMatchedPairs();
        }
    }
    const juce::ScopedLock l(dataLock);
    loaded = false;
    song = std::move(b);
    songRate = r->sampleRate;
    midiSeq = seq;
    loaded = true;
}

void AIMusicProcessor::processBlock(juce::AudioBuffer<float>& buf, juce::MidiBuffer& midi)
{
    buf.clear();
    midi.clear();
    juce::AudioPlayHead::PositionInfo pos;
    if (auto* ph = getPlayHead()) if (auto p = ph->getPosition()) pos = *p;
    const bool playing = pos.getIsPlaying();
    const double t0 = pos.getTimeInSeconds().orFallback(0.0);
    if (!playing || !loaded.load()) { wasPlaying = playing; return; }

    const juce::ScopedTryLock tl(dataLock);
    if (!tl.isLocked()) return;
    const int n = buf.getNumSamples();
    const double ratio = songRate / hostRate;
    const double startSrc = t0 * songRate;
    for (int i = 0; i < n; ++i)
    {
        const double s = startSrc + i * ratio;
        const int i0 = (int) s;
        if (s < 0 || i0 + 1 >= song.getNumSamples()) continue;
        const float f = (float) (s - i0);
        for (int c = 0; c < juce::jmin(2, buf.getNumChannels()); ++c)
            buf.setSample(c, i, song.getSample(c, i0) * (1.f - f) + song.getSample(c, i0 + 1) * f);
    }
    // MIDI events falling inside this block
    const double t1 = t0 + n / hostRate;
    for (int e = 0; e < midiSeq.getNumEvents(); ++e)
    {
        auto* ev = midiSeq.getEventPointer(e);
        const double ts = ev->message.getTimeStamp();
        if (ts >= t0 && ts < t1 && !ev->message.isMetaEvent())
            midi.addEvent(ev->message, juce::jlimit(0, n - 1, (int) ((ts - t0) * hostRate)));
    }
    wasPlaying = playing;
}

juce::AudioProcessorEditor* AIMusicProcessor::createEditor() { return new AIMusicEditor(*this); }

void AIMusicProcessor::getStateInformation(juce::MemoryBlock& d)
{
    juce::ValueTree v("AIMG");
    v.setProperty("engine", enginePath, nullptr); v.setProperty("vocal", vocalPath, nullptr);
    v.setProperty("genre", genre, nullptr); v.setProperty("sf", soundfontPath, nullptr); v.setProperty("bars", bars, nullptr);
    juce::MemoryOutputStream o(d, false);
    v.writeToStream(o);
}
void AIMusicProcessor::setStateInformation(const void* d, int n)
{
    auto v = juce::ValueTree::readFromData(d, (size_t) n);
    if (!v.isValid()) return;
    enginePath = v["engine"]; vocalPath = v["vocal"]; genre = v["genre"].toString();
    soundfontPath = v["sf"].toString(); bars = (int) v["bars"];
    if (genre.isEmpty()) genre = "isan";
    if (bars <= 0) bars = 32;
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new AIMusicProcessor(); }
