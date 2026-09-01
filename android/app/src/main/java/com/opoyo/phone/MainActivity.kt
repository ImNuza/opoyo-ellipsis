package com.opoyo.phone

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.log10
import kotlin.math.sqrt

class MainActivity : AppCompatActivity(), SensorEventListener {

    private val labels = listOf("heeldrop", "bag", "book", "pan", "door", "walk", "quiet", "tv")
    private val g = 9.81f
    private val capturing = AtomicBoolean(false)
    private val rows = ArrayList<String>(4096)
    private val main = Handler(Looper.getMainLooper())

    private lateinit var sensors: SensorManager
    private var linear: Sensor? = null
    private var recorder: AudioRecord? = null
    private var audioThread: Thread? = null
    @Volatile private var db = -120.0
    @Volatile private var magG = 0.0
    private var lastFile: File? = null

    private lateinit var readout: TextView
    private lateinit var status: TextView
    private lateinit var label: Spinner
    private lateinit var record: Button
    private lateinit var send: Button

    private val askMic = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { ok ->
        if (ok) startTake() else status.text = "Mic permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        readout = findViewById(R.id.readout)
        status = findViewById(R.id.status)
        label = findViewById(R.id.label)
        record = findViewById(R.id.record)
        send = findViewById(R.id.send)

        label.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, labels)
        sensors = getSystemService(SENSOR_SERVICE) as SensorManager
        linear = sensors.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

        record.setOnClickListener {
            if (capturing.get()) stopTake() else maybeStart()
        }
        send.setOnClickListener { share() }
        main.post(tick)
    }

    private val tick = object : Runnable {
        override fun run() {
            readout.text = String.format(Locale.US, "mag %.3f g   db %.1f", magG, db)
            if (capturing.get()) {
                status.text = "Recording ${currentLabel()} · ${rows.size} samples"
            }
            main.postDelayed(this, 200)
        }
    }

    private fun currentLabel(): String = labels.getOrElse(label.selectedItemPosition) { "heeldrop" }

    private fun maybeStart() {
        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (granted) startTake() else askMic.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun startTake() {
        if (linear == null) {
            status.text = "No linear accelerometer"
            return
        }
        rows.clear()
        lastFile = null
        send.isEnabled = false
        capturing.set(true)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        sensors.registerListener(this, linear, SensorManager.SENSOR_DELAY_GAME)
        startMic()
        record.text = "Stop take"
        label.isEnabled = false
        status.text = "Recording ${currentLabel()}…"
    }

    private fun stopTake() {
        capturing.set(false)
        sensors.unregisterListener(this)
        stopMic()
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        record.text = "Record ${currentLabel()}"
        label.isEnabled = true
        val file = writeCsv()
        lastFile = file
        if (file == null) {
            status.text = "No samples — try again"
            return
        }
        send.isEnabled = true
        status.text = "Saved ${file.name}. Send it."
        send.text = "Send ${file.name}"
    }

    private fun writeCsv(): File? {
        if (rows.isEmpty()) return null
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(cacheDir, "takes").apply { mkdirs() }
        val file = File(dir, "${currentLabel()}_${stamp}.csv")
        file.writeText("t,ax,ay,az,mag,db\n" + rows.joinToString(""))
        return file
    }

    private fun share() {
        val file = lastFile ?: return
        val uri = FileProvider.getUriForFile(this, "$packageName.files", file)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/csv"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, file.name)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "Send take"))
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_LINEAR_ACCELERATION) return
        val ax = event.values[0] / g
        val ay = event.values[1] / g
        val az = event.values[2] / g
        magG = sqrt((ax * ax + ay * ay + az * az).toDouble())
        if (!capturing.get()) return
        val t = System.currentTimeMillis()
        rows.add(
            String.format(Locale.US, "%d,%.4f,%.4f,%.4f,%.4f,%.2f\n", t, ax, ay, az, magG, db)
        )
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun startMic() {
        val rate = 44100
        val min = AudioRecord.getMinBufferSize(
            rate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            rate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            min * 2
        )
        recorder = rec
        rec.startRecording()
        val running = capturing
        audioThread = Thread {
            val buf = ShortArray(1024)
            while (running.get()) {
                val n = rec.read(buf, 0, buf.size)
                if (n <= 0) continue
                var sum = 0.0
                for (i in 0 until n) {
                    val v = buf[i] / 32768.0
                    sum += v * v
                }
                val rms = sqrt(sum / n)
                db = 20.0 * log10(maxOf(rms, 1e-8))
            }
        }.also { it.start() }
    }

    private fun stopMic() {
        capturing.set(false)
        audioThread?.join(400)
        audioThread = null
        recorder?.stop()
        recorder?.release()
        recorder = null
    }

    override fun onDestroy() {
        if (capturing.get()) stopTake()
        main.removeCallbacksAndMessages(null)
        super.onDestroy()
    }
}
