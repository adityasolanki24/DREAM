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

    private fun encrypt(value: String): String {
        val c = Cipher.getInstance("AES/GCM/NoPadding")
        c.init(Cipher.ENCRYPT_MODE, key())
        return Base64.encodeToString(c.iv + c.doFinal(value.toByteArray(Charsets.UTF_8)), Base64.NO_WRAP)
    }

    private fun decrypt(blob: String): String? = runCatching {
        val raw = Base64.decode(blob, Base64.NO_WRAP)
        val c = Cipher.getInstance("AES/GCM/NoPadding")
        c.init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, raw, 0, 12))
        String(c.doFinal(raw, 12, raw.size - 12), Charsets.UTF_8)
    }.getOrNull()

    /* ---- connection profiles: one per robot / backend; tokens are per profile and Keystore-encrypted ---- */

    init { migrate() }

    /** Older builds kept one endpoint and one token; they become the first profile. */
    private fun migrate() {
        if (prefs.contains("profiles")) return
        val legacy = ConnectionProfile("usb", "This laptop (USB)", prefs.getString("endpoint", DEFAULT_ENDPOINT) ?: DEFAULT_ENDPOINT)
        val e = prefs.edit().putString("profiles", org.json.JSONArray().put(legacy.toJson()).toString()).putString("active", legacy.id)
        prefs.getString("token", null)?.let { e.putString("token:${legacy.id}", it).remove("token") }
        e.remove("endpoint").apply()
    }

    var profiles: List<ConnectionProfile>
        get() = runCatching { org.json.JSONArray(prefs.getString("profiles", "[]")).objects().map { ConnectionProfile.parse(it) } }.getOrDefault(emptyList())
        set(v) = prefs.edit().putString("profiles", org.json.JSONArray().apply { v.forEach { put(it.toJson()) } }.toString()).apply()

    var activeId: String?
        get() = prefs.getString("active", null)?.takeIf { id -> profiles.any { it.id == id } } ?: profiles.firstOrNull()?.id
        set(v) = prefs.edit().putString("active", v).apply()

    val active: ConnectionProfile? get() = profiles.find { it.id == activeId }

    fun tokenFor(id: String): String? = prefs.getString("token:$id", null)?.let { decrypt(it) }
    fun setToken(id: String, token: String?) {
        if (token.isNullOrBlank()) prefs.edit().remove("token:$id").apply() else prefs.edit().putString("token:$id", encrypt(token)).apply()
    }

    /** Adds or replaces a profile (matched by id). */
    fun save(profile: ConnectionProfile) { profiles = profiles.filterNot { it.id == profile.id } + profile }

    fun remove(id: String) {
        profiles = profiles.filterNot { it.id == id }
        setToken(id, null)
        if (prefs.getString("active", null) == id) activeId = profiles.firstOrNull()?.id
    }

    /** The active profile's token and endpoint (what the session connects with). */
    val token: String? get() = activeId?.let { tokenFor(it) }
    val endpoint: String get() = active?.endpoint ?: DEFAULT_ENDPOINT

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
