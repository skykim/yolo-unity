using System;
using System.Collections.Generic;
using System.IO;
using Unity.InferenceEngine;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using Unity.Mathematics;

// instead of the Sentis Functional.NMS layer.
public class RunYOLOGPU : MonoBehaviour
{
    [Tooltip("Drag a YOLO model .onnx file here")]
    public ModelAsset modelAsset;

    [Tooltip("Drag the classes.txt here")]
    public TextAsset classesAsset;

    [Tooltip("Create a Raw Image in the scene and link it here")]
    public RawImage displayImage;

    [Tooltip("Drag a border box texture here")]
    public Texture2D borderTexture;

    [Tooltip("Select an appropriate font for the labels")]
    public Font font;

    // Compute Shader for NMS
    [Tooltip("Drag your NMS Compute Shader file here (NMSCompute.compute)")]
    public ComputeShader nmsComputeShader;

    [Tooltip("Change this to the name of the video you put in the Assets/StreamingAssets folder")]
    public string videoFilename = "giraffes.mp4";

    const BackendType backend = BackendType.GPUCompute;

    private Transform displayLocation;
    private Worker worker;
    private string[] labels;
    private RenderTexture targetRT;
    private Sprite borderSprite;

    //Image size for the model
    private const int imageWidth = 640;
    private const int imageHeight = 640;

    private VideoPlayer video;

    List<GameObject> boxPool = new();

    [Tooltip("Intersection over union threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)]
    float iouThreshold = 0.5f;

    [Tooltip("Confidence score threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)]
    float scoreThreshold = 0.5f;

    Tensor<float> centersToCorners;

    // Total number of candidates from YOLO output (e.g., 8400 for 640x640)
    private const int numCandidates = 8400;
    // Maximum number of boxes to detect
    private const int maxBoxes = 200;

    // Kernel ID for the NMS Compute Shader
    private int nmsKernel;

    // GPU buffer for final box coordinates (AppendBuffer)
    private ComputeBuffer outCoordsGpu;
    // GPU buffer for final label IDs (AppendBuffer)
    private ComputeBuffer outLabelIDsGpu;
    // GPU buffer to hold the count from AppendBuffers
    private ComputeBuffer countGpu;

    // CPU array to read back the count (must be 16-bytes aligned, hence int[4])
    private int[] countArray = new int[4];
    // CPU array to read back coordinates (using Vector4 for 16-byte alignment)
    private Vector4[] finalCoordsArray;
    // CPU array to read back label IDs (using Vector4 for 16-byte alignment)
    private Vector4[] finalLabelsArray;

    //bounding box data
    public struct BoundingBox
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
        public string label;
    }

    void Start()
    {
        Application.targetFrameRate = 60;
        Screen.orientation = ScreenOrientation.LandscapeLeft;

        labels = classesAsset.text.Split('\n');

        LoadModel();

        targetRT = new RenderTexture(imageWidth, imageHeight, 0);

        displayLocation = displayImage.transform;

        SetupInput();

        borderSprite = Sprite.Create(borderTexture, new Rect(0, 0, borderTexture.width, borderTexture.height), new Vector2(borderTexture.width / 2, borderTexture.height / 2));
    }
    void LoadModel()
    {
        var model1 = ModelLoader.Load(modelAsset);

        centersToCorners = new Tensor<float>(new TensorShape(4, 4),
        new float[]
        {
                    1,      0,      1,      0,
                    0,      1,      0,      1,
                    -0.5f,  0,      0.5f,   0,
                    0,      -0.5f,  0,      0.5f
        });

        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model1);
        var modelOutput = Functional.Forward(model1, inputs)[0];
        var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);
        var allScores = modelOutput[0, 4.., ..];
        var scores = Functional.ReduceMax(allScores, 0);
        var classIDs = Functional.ArgMax(allScores, 0);
        var boxCorners = Functional.MatMul(boxCoords, Functional.Constant(centersToCorners));

        // Compile the graph to output 4 tensors directly (no NMS)
        worker = new Worker(graph.Compile(boxCoords, classIDs, boxCorners, scores), backend);

        // Find the NMS kernel in the Compute Shader
        nmsKernel = nmsComputeShader.FindKernel("RunNMS");

        // Initialize AppendBuffer for coordinates (float4 = 16 bytes)
        outCoordsGpu = new ComputeBuffer(maxBoxes, sizeof(float) * 4, ComputeBufferType.Append);
        // Initialize AppendBuffer for labels (int4 = 16 bytes for alignment)
        outLabelIDsGpu = new ComputeBuffer(maxBoxes, sizeof(int) * 4, ComputeBufferType.Append);
        // Initialize counter buffer (must be 16 bytes total)
        countGpu = new ComputeBuffer(4, sizeof(int), ComputeBufferType.Raw);

        // Initialize CPU array for coordinates (Vector4 for 16-byte alignment)
        finalCoordsArray = new Vector4[maxBoxes];
        // Initialize CPU array for labels (Vector4 for 16-byte alignment)
        finalLabelsArray = new Vector4[maxBoxes];
    }

    void SetupInput()
    {
        video = gameObject.AddComponent<VideoPlayer>();
        video.renderMode = VideoRenderMode.APIOnly;
        video.source = VideoSource.Url;
        video.url = Path.Join(Application.streamingAssetsPath, videoFilename);
        video.isLooping = true;
        video.Play();
    }

    private void Update()
    {
        ExecuteML();

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }
    }

    public void ExecuteML()
    {
        ClearAnnotations();

        if (video && video.texture)
        {
            float aspect = video.width * 1f / video.height;
            Graphics.Blit(video.texture, targetRT, new Vector2(1f / aspect, 1), new Vector2(0, 0));
            displayImage.texture = targetRT;
        }
        else
        {
            return;
        }

        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));
        TextureConverter.ToTensor(targetRT, inputTensor, default);

        worker.Schedule(inputTensor);

        // Peek model outputs (GPU handles) without downloading data
        var boxCoordsTensor = worker.PeekOutput("output_0") as Tensor<float>;
        var classIDsTensor = worker.PeekOutput("output_1") as Tensor<int>;
        var boxCornersTensor = worker.PeekOutput("output_2") as Tensor<float>;
        var scoresTensor = worker.PeekOutput("output_3") as Tensor<float>;

        // Pin tensors to get their underlying ComputeBuffer
        var boxCoordsPin = ComputeTensorData.Pin(boxCoordsTensor);
        var classIDsPin = ComputeTensorData.Pin(classIDsTensor);
        var boxCornersPin = ComputeTensorData.Pin(boxCornersTensor);
        var scoresPin = ComputeTensorData.Pin(scoresTensor);

        // Reset AppendBuffer counters
        outCoordsGpu.SetCounterValue(0);
        outLabelIDsGpu.SetCounterValue(0);

        // Set input/output buffers for the NMS Compute Shader
        nmsComputeShader.SetBuffer(nmsKernel, "inBoxCoords", boxCoordsPin.buffer);
        nmsComputeShader.SetBuffer(nmsKernel, "inClassIDs", classIDsPin.buffer);
        nmsComputeShader.SetBuffer(nmsKernel, "inBoxCorners", boxCornersPin.buffer);
        nmsComputeShader.SetBuffer(nmsKernel, "inScores", scoresPin.buffer);
        nmsComputeShader.SetBuffer(nmsKernel, "outCoords", outCoordsGpu);
        nmsComputeShader.SetBuffer(nmsKernel, "outLabelIDs", outLabelIDsGpu);
        // Set NMS parameters
        nmsComputeShader.SetFloat("scoreThreshold", scoreThreshold);
        nmsComputeShader.SetFloat("iouThreshold", iouThreshold);

        // Run the NMS Compute Shader on the GPU
        nmsComputeShader.Dispatch(nmsKernel, (numCandidates + 63) / 64, 1, 1);

        // Copy the AppendBuffer count to the counter buffer (GPU-side)
        ComputeBuffer.CopyCount(outCoordsGpu, countGpu, 0);

        // Read back only the count (16 bytes) from GPU to CPU
        countGpu.GetData(countArray);

        // The actual count is the first element of the aligned array
        int boxesFound = countArray[0];
        if (boxesFound > maxBoxes) boxesFound = maxBoxes;

        // Read back only the final coordinate data
        outCoordsGpu.GetData(finalCoordsArray, 0, 0, boxesFound);
        // Read back only the final label data
        outLabelIDsGpu.GetData(finalLabelsArray, 0, 0, boxesFound);

        float displayWidth = displayImage.rectTransform.rect.width;
        float displayHeight = displayImage.rectTransform.rect.height;

        float scaleX = displayWidth / imageWidth;
        float scaleY = displayHeight / imageHeight;

        for (int n = 0; n < boxesFound; n++)
        {
            var box = new BoundingBox
            {
                // Read coordinate data from the Vector4 array (.x, .y, .z, .w)
                centerX = finalCoordsArray[n].x * scaleX - displayWidth / 2,
                centerY = finalCoordsArray[n].y * scaleY - displayHeight / 2,
                width = finalCoordsArray[n].z * scaleX,
                height = finalCoordsArray[n].w * scaleY,

                // Reinterpret the bits of the float (.x) as an int
                label = labels[math.asint(finalLabelsArray[n].x)],
            };
            DrawBox(box, n, displayHeight * 0.05f);
        }
    }

    public void DrawBox(BoundingBox box, int id, float fontSize)
    {
        GameObject panel;
        if (id < boxPool.Count)
        {
            panel = boxPool[id];
            panel.SetActive(true);
        }
        else
        {
            panel = CreateNewBox(Color.yellow);
        }
        panel.transform.localPosition = new Vector3(box.centerX, -box.centerY);

        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(box.width, box.height);

        var label = panel.GetComponentInChildren<Text>();
        label.text = box.label;
        label.fontSize = (int)fontSize;
    }

    public GameObject CreateNewBox(Color color)
    {
        var panel = new GameObject("ObjectBox");
        panel.AddComponent<CanvasRenderer>();
        Image img = panel.AddComponent<Image>();
        img.color = color;
        img.sprite = borderSprite;
        img.type = Image.Type.Sliced;
        panel.transform.SetParent(displayLocation, false);

        var text = new GameObject("ObjectLabel");
        text.AddComponent<CanvasRenderer>();
        text.transform.SetParent(panel.transform, false);
        Text txt = text.AddComponent<Text>();
        txt.font = font;
        txt.color = color;
        txt.fontSize = 40;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;

        RectTransform rt2 = text.GetComponent<RectTransform>();
        rt2.offsetMin = new Vector2(20, rt2.offsetMin.y);
        rt2.offsetMax = new Vector2(0, rt2.offsetMax.y);
        rt2.offsetMin = new Vector2(rt2.offsetMin.x, 0);
        rt2.offsetMax = new Vector2(rt2.offsetMax.x, 30);
        rt2.anchorMin = new Vector2(0, 0);
        rt2.anchorMax = new Vector2(1, 1);

        boxPool.Add(panel);
        return panel;
    }

    public void ClearAnnotations()
    {
        foreach (var box in boxPool)
        {
            box.SetActive(false);
        }
    }

    void OnDestroy()
    {
        centersToCorners?.Dispose();
        worker?.Dispose();

        // Release all Compute Buffers
        outCoordsGpu?.Dispose();
        outLabelIDsGpu?.Dispose();
        countGpu?.Dispose();
    }
}