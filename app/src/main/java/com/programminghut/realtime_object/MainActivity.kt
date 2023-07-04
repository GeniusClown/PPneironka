package com.programminghut.realtime_object

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.speech.tts.TextToSpeech
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.core.content.ContextCompat
import com.programminghut.realtime_object.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import android.widget.TextView
import java.util.*
import kotlin.concurrent.schedule

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    lateinit var labels: List<String>
    var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED
    )
    val paint = Paint()
    val spokenItems = mutableListOf<String>()
    val speakRunnable = Runnable {
        spokenItems.clear() // Очистить список озвученных предметов после задержки
    }
    lateinit var imageProcessor: ImageProcessor
    lateinit var bitmap: Bitmap
    lateinit var imageView: ImageView
    lateinit var cameraDevice: CameraDevice
    lateinit var handler: Handler
    lateinit var cameraManager: CameraManager
    lateinit var textureView: TextureView
    lateinit var model: SsdMobilenetV11Metadata1
    lateinit var textToSpeech: TextToSpeech

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        get_permission()

        labels = FileUtil.loadLabels(this, "labels.txt")
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()
        model = SsdMobilenetV11Metadata1.newInstance(this)
        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        imageView = findViewById(R.id.imageView)
        textureView = findViewById(R.id.textureView)
        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                open_camera()
            }

            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {}

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                bitmap = textureView.bitmap!!
                var image = TensorImage.fromBitmap(bitmap)
                image = imageProcessor.process(image)
                val outputs = model.process(image)
                val locations = outputs.locationsAsTensorBuffer.floatArray
                val classes = outputs.classesAsTensorBuffer.floatArray
                val scores = outputs.scoresAsTensorBuffer.floatArray
                val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray
                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                // Подсчет объектов
                val desiredLabels = listOf("pen", "pencil", "mouse", "keyboard")
                var objectCount = 0
                for (i in 0 until numberOfDetections.size) {
                    val score = scores[i]
                    val label = labels[classes[i].toInt()]
                    if (score > 0.5 && label in desiredLabels) {
                        objectCount++
                    }
                }

                // Отобразить результаты подсчета объектов
                runOnUiThread {
                    val countTextView = findViewById<TextView>(R.id.countTextView)
                    countTextView.text = "Количество объектов: " + objectCount.toString()
                }
                val h = mutable.height
                val w = mutable.width
                paint.textSize = h / 15f
                paint.strokeWidth = h / 85f
                var x = 0
                scores.forEachIndexed { index, fl ->
                    x = index
                    x *= 4
                    val label = labels[classes[index].toInt()]
                    if (fl > 0.5 && label in desiredLabels && label !in spokenItems) {
                        paint.setColor(colors[index % colors.size])
                        paint.style = Paint.Style.STROKE
                        canvas.drawRect(
                            RectF(
                                locations[x + 1] * w,
                                locations[x] * h,
                                locations[x + 3] * w,
                                locations[x + 2] * h
                            ),
                            paint
                        )
                        paint.style = Paint.Style.FILL
                        canvas.drawText(
                            "$label ${fl.toString()}",
                            locations[x + 1] * w,
                            locations[x] * h,
                            paint
                        )
                        textToSpeech.speak(label, TextToSpeech.QUEUE_ADD, null, null)
                        spokenItems.add(label) // Добавьте озвученный предмет в список

                        // Запустите задержку на 1 секунду для повторного озвучивания
                        handler.postDelayed(speakRunnable, 1500)
                    }
                }

                imageView.setImageBitmap(mutable)
            }
        }

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

        // Инициализация TextToSpeech
        textToSpeech = TextToSpeech(this, this)
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
        textToSpeech.shutdown() // Освобождение ресурсов TextToSpeech
    }

    @SuppressLint("MissingPermission")
    fun open_camera() {
        cameraManager.openCamera(cameraManager.cameraIdList[0], object : CameraDevice.StateCallback() {
            override fun onOpened(p0: CameraDevice) {
                cameraDevice = p0

                var surfaceTexture = textureView.surfaceTexture
                var surface = Surface(surfaceTexture)

                var captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                cameraDevice.createCaptureSession(
                    listOf(surface),
                    object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(p0: CameraCaptureSession) {
                            p0.setRepeatingRequest(captureRequest.build(), null, null)
                        }

                        override fun onConfigureFailed(p0: CameraCaptureSession) {}
                    },
                    handler
                )
            }

            override fun onDisconnected(p0: CameraDevice) {}

            override fun onError(p0: CameraDevice, p1: Int) {}
        }, handler)
    }

    fun get_permission() {
        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            // Установить язык на английский, если требуемый язык не доступен
            val result = textToSpeech.setLanguage(Locale.getDefault())
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                textToSpeech.language = Locale.US
            }
        }
    }
}
