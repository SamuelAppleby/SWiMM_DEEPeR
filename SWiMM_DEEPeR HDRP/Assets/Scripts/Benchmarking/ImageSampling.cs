using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;

public class ImageSampling : MonoBehaviour
{
    private GameObject target_obj;
    public Camera camera_obj1;
    public Camera camera_obj2;
    public int degree_change = 1;
    public int radius = 1;
    private List<Tuple<int, int>> resolutions;
    private string image_dir = "../benchmarking/images/";
    public GameObject prefab;
    public Transform screenshot_zone;
    private string csv_dir = "../benchmarking/";
    private string graphics_pipeline;
    private int num_samples = 5;
    private bool take_screenshot = false;

    void Start()
    {
        graphics_pipeline = GraphicsSettings.defaultRenderPipeline == null ? "built_in" : "hdrp";
        image_dir += graphics_pipeline + "/";

#if !UNITY_EDITOR
        image_dir = "../../" + image_dir;
        csv_dir = "../../" + csv_dir;
#endif

        resolutions = new List<Tuple<int, int>>
        {
            new Tuple<int, int>(32,32),
            new Tuple<int, int>(64,64),
            //new Tuple<int, int>(128,128),
            //new Tuple<int, int>(256,256),
            //new Tuple<int, int>(512,512),
            //new Tuple<int, int>(1024,1024),
            //new Tuple<int, int>(1920,1080)
        };

    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.KeypadEnter))
        {
            Utils.CleanAndCreateDirectories(new Dictionary<string, bool>()
            {
                { image_dir, true },
                { csv_dir, false }
            });

            foreach (Tuple<int, int> res in resolutions)
            {
                string res_path = image_dir + res.Item1.ToString() + "x" + res.Item2.ToString() + "/";

                Utils.CleanAndCreateDirectories(new Dictionary<string, bool>()
                {
                    {res_path, true }
                });
            }

            /* Turn off buoyancy during screenshotting */
            foreach (Floater floater in GetComponentsInChildren<Floater>())
            {
                floater.buoyant_strength = 0;
            }

            transform.rotation = Quaternion.identity;
            target_obj = Instantiate(prefab, screenshot_zone.position, Quaternion.identity);

            take_screenshot = true;
            //StartCoroutine(TakeScreenshot());

            /* Turn off buoyancy during screenshotting */
            foreach (Floater floater in GetComponentsInChildren<Floater>())
            {
                floater.buoyant_strength = GetComponent<FloaterContainer>().submerged_buoyant_strength / GetComponent<FloaterContainer>().num_floaters;
            }
        }
    }

    private void OnEnable()
    {
        RenderPipelineManager.endCameraRendering += RenderPipelineManager_endCameraRendering;
    }

    private void OnDisable()
    {
        RenderPipelineManager.endCameraRendering -= RenderPipelineManager_endCameraRendering;
    }

    private void RenderPipelineManager_endCameraRendering(ScriptableRenderContext arg1, Camera arg2)
    {
        if (take_screenshot)
        {
            var csv = new StringBuilder();

            for (int degree = 1; degree < 2; ++degree)
            {
                float opposite = Mathf.Sin((degree * Mathf.PI) / 180) * radius;
                float adjacent = Mathf.Cos((degree * Mathf.PI) / 180) * radius;

                transform.position = target_obj.transform.position + new Vector3(opposite, 0, -adjacent);
                transform.Rotate(-Vector3.up);

                foreach (Tuple<int, int> res in resolutions)
                {
                    string res_path = image_dir + res.Item1.ToString() + "x" + res.Item2.ToString() + "/";

                    /* Benchmark 5 times for behind image */
                    for (int i = degree == 0 ? 0 : num_samples; i < num_samples + 1; ++i)
                    {
                        float start_time = Time.realtimeSinceStartup;

                        if(res.Item1 == 32)
                        {
                            RenderTexture.active = camera_obj1.targetTexture;
                        }
                        else
                        {
                            RenderTexture.active = camera_obj2.targetTexture;
                        }
                        Texture2D screen_shot = new Texture2D(res.Item1, res.Item2, TextureFormat.RGB24, false);
                        screen_shot.ReadPixels(new Rect(0, 0, res.Item1, res.Item2), 0, 0);
                        screen_shot.Apply();

                        //camera_obj.targetTexture = null;
                        //RenderTexture.active = null;
                        //Destroy(rt);
                        Destroy(screen_shot);

                        File.WriteAllBytes(res_path + "image_" + degree + ".jpg", screen_shot.EncodeToJPG());

                        /* Time to record time taken */
                        if (degree == 0)
                        {
                            if (i == 0)
                            {
                                continue;   // Prioritising caching of the OS
                            }

                            float time_taken = Time.realtimeSinceStartup - start_time;
                            var newLine = $"{res.Item1.ToString() + "x" + res.Item2.ToString()},{graphics_pipeline},{time_taken}";
                            csv.AppendLine(newLine);
                        }
                    }
                }
            }

            File.AppendAllText(csv_dir + "results.csv", csv.ToString());
            Destroy(target_obj);

            take_screenshot = false;
        }
    }

    //private IEnumerator TakeScreenshot()
    //{
    //    yield return new WaitForEndOfFrame();

    //    var csv = new StringBuilder();

    //    for (int degree = 0; degree < 1; ++degree)
    //    {
    //        float opposite = Mathf.Sin((degree * Mathf.PI) / 180) * radius;
    //        float adjacent = Mathf.Cos((degree * Mathf.PI) / 180) * radius;

    //        transform.position = target_obj.transform.position + new Vector3(opposite, 0, -adjacent);
    //        transform.Rotate(-Vector3.up);

    //        foreach (Tuple<int, int> res in resolutions)
    //        {
    //            string res_path = image_dir + res.Item1.ToString() + "x" + res.Item2.ToString() + "/";

    //            /* Benchmark 5 times for behind image */
    //            for(int i = degree == 0 ? 0 : num_samples; i < num_samples + 1; ++i) {
    //                float start_time = Time.realtimeSinceStartup;


    //                Texture2D screen_shot = new Texture2D(res.Item1, res.Item2, TextureFormat.RGB24, false);
    //                screen_shot.ReadPixels(new Rect(0, 0, res.Item1, res.Item2), 0, 0);
    //                screen_shot.Apply();

    //                camera_obj.targetTexture = null;
    //                RenderTexture.active = null;
    //                Destroy(rt);
    //                Destroy(screen_shot);

    //                File.WriteAllBytes(res_path + "image_" + degree + ".jpg", screen_shot.EncodeToJPG());

    //                /* Time to record time taken */
    //                if (degree == 0)
    //                {
    //                    if(i== 0)
    //                    {
    //                        continue;   // Prioritising caching of the OS
    //                    }

    //                    float time_taken = Time.realtimeSinceStartup - start_time;
    //                    var newLine = $"{res.Item1.ToString() + "x" + res.Item2.ToString()},{graphics_pipeline},{time_taken}";
    //                    csv.AppendLine(newLine);
    //                }
    //            }
    //        }
    //    }

    //    File.AppendAllText(csv_dir + "results.csv", csv.ToString());
    //    Destroy(target_obj);
    //}
}
