package dev.dream.companion.data

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * The DREAM access token, encrypted with an AES-GCM key that never leaves the Android Keystore (API 23+).
 * Non-secret preferences (endpoint, chosen model, toggles) live in ordinary SharedPreferences.
 */
class CredentialStore(context: Context) {
    private val prefs = context.getSharedPreferences("dream", Context.MODE_PRIVATE)

    private fun key(): SecretKey {
        val ks = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (ks.getKey(ALIAS, null) as? SecretKey)?.let { return it }
        val gen = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        gen.init(
            KeyGenParameterSpec.Builder(ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).setKeySize(256).build(),
        )
        return gen.generateKey()
    }

    var token: String?
        get() = runCatching {
            val blob = prefs.getString("token", null) ?: return null
            val raw = Base64.decode(blob, Base64.NO_WRAP)
            val c = Cipher.getInstance("AES/GCM/NoPadding")
            c.init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, raw, 0, 12))
            String(c.doFinal(raw, 12, raw.size - 12), Charsets.UTF_8)
        }.getOrNull()
        set(value) {
            if (value.isNullOrBlank()) { prefs.edit().remove("token").apply(); return }
            val c = Cipher.getInstance("AES/GCM/NoPadding")
            c.init(Cipher.ENCRYPT_MODE, key())
            val out = c.iv + c.doFinal(value.toByteArray(Charsets.UTF_8))
            prefs.edit().putString("token", Base64.encodeToString(out, Base64.NO_WRAP)).apply()
        }

    var endpoint: String
        get() = prefs.getString("endpoint", DEFAULT_ENDPOINT) ?: DEFAULT_ENDPOINT
        set(v) = prefs.edit().putString("endpoint", v.trim()).apply()

    var choice: ModelChoice?
        get() = prefs.getString("choice_provider", null)?.let { ModelChoice(it, prefs.getString("choice_model", null)) }
        set(v) = prefs.edit().putString("choice_provider", v?.provider).putString("choice_model", v?.model).apply()

    var speakReplies: Boolean
        get() = prefs.getBoolean("speak", false)
        set(v) = prefs.edit().putBoolean("speak", v).apply()

    var reducedMotion: Boolean
        get() = prefs.getBoolean("reduced_motion", false)
        set(v) = prefs.edit().putBoolean("reduced_motion", v).apply()

    companion object {
        private const val ALIAS = "dream-token-key"
        /** DREAM's default phone API port, reached from the phone through `adb reverse tcp:8765 tcp:8765`. */
        const val DEFAULT_ENDPOINT = "http://127.0.0.1:8765"
    }
}
