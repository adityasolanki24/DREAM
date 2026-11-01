package dev.dream.companion.audio

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.Locale

enum class MicState { IDLE, STARTING, LISTENING, PROCESSING, UNAVAILABLE, ERROR }

/**
 * Push-to-talk speech-to-text with Android's SpeechRecognizer (on this phone: Google's recognition service).
 * It only ever runs while the user is holding (or has tapped on) the microphone; results go into the composer and are
 * never submitted as a robot command without the user's send action.
 */
class Voice(private val context: Context) {
    private val _state = MutableStateFlow(if (SpeechRecognizer.isRecognitionAvailable(context)) MicState.IDLE else MicState.UNAVAILABLE)
    val state: StateFlow<MicState> = _state
    private val _partial = MutableStateFlow("")
    val partial: StateFlow<String> = _partial
    private val _error = MutableStateFlow("")
    val error: StateFlow<String> = _error
    private var recognizer: SpeechRecognizer? = null
    private var onFinal: ((String) -> Unit)? = null
    val available get() = SpeechRecognizer.isRecognitionAvailable(context)

    fun start(onResult: (String) -> Unit) {
        if (!available) { _state.value = MicState.UNAVAILABLE; return }
        cancel()
        onFinal = onResult
        _partial.value = ""; _error.value = ""
        val r = SpeechRecognizer.createSpeechRecognizer(context)
        recognizer = r
        r.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) { _state.value = MicState.LISTENING }
            override fun onBeginningOfSpeech() { _state.value = MicState.LISTENING }
            override fun onRmsChanged(rmsdB: Float) { level.value = ((rmsdB + 2f) / 12f).coerceIn(0f, 1f) }
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() { _state.value = MicState.PROCESSING; level.value = 0f }
            override fun onError(error: Int) {
                level.value = 0f
                _error.value = describe(error)
                _state.value = if (error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) MicState.IDLE else MicState.ERROR
                release()
            }
            override fun onResults(results: Bundle?) {
                val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull().orEmpty()
                _partial.value = ""
                if (text.isNotBlank()) onFinal?.invoke(text)
                _state.value = MicState.IDLE
                release()
            }
            override fun onPartialResults(partialResults: Bundle?) {
                _partial.value = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull().orEmpty()
            }
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault().toLanguageTag())
            .putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            .putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        _state.value = MicState.STARTING
        r.startListening(intent)
    }

    /** Release of push-to-talk: stop capturing and let the recognizer return what it heard. */
    fun stop() { if (_state.value == MicState.LISTENING || _state.value == MicState.STARTING) { _state.value = MicState.PROCESSING; recognizer?.stopListening() } }

    fun clearError() { _error.value = "" }

    fun cancel() { recognizer?.cancel(); release(); if (_state.value != MicState.UNAVAILABLE) _state.value = MicState.IDLE }

    private fun release() { recognizer?.destroy(); recognizer = null }

    val level = MutableStateFlow(0f)

    private fun describe(code: Int) = when (code) {
        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Microphone permission is required."
        SpeechRecognizer.ERROR_NETWORK, SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "The speech service needs a network connection."
        SpeechRecognizer.ERROR_NO_MATCH -> "No speech was recognised. Try again."
        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "No speech heard."
        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "The speech service is busy."
        SpeechRecognizer.ERROR_AUDIO -> "The microphone could not be read."
        else -> "Speech recognition failed (code $code)."
    }
}

/** Optional spoken replies with the platform TextToSpeech engine (Google TTS on this phone). */
class Speaker(context: Context) {
    private var ready = false
    private val tts: TextToSpeech = TextToSpeech(context.applicationContext) { status -> ready = status == TextToSpeech.SUCCESS; if (ready) setLocale() }
    private fun setLocale() { runCatching { tts.language = Locale.getDefault() } }
    val isReady get() = ready
    fun say(text: String) { if (ready && text.isNotBlank()) tts.speak(text.take(280), TextToSpeech.QUEUE_FLUSH, null, "dream-${text.hashCode()}") }
    fun stop() { if (ready) tts.stop() }
}
