using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;

public class ImageSampling : MonoBehaviour
{
    [SerializeField]
    private GameObject target_prefab;

    [HideInInspector]
    private Transform target_trans;

    [SerializeField]
    private Camera track_camera;

    private string image_dir;

    private string data_dir;

    [SerializeField]
    private float distance;

    [SerializeField]
    private List<Resolution> resolutions;

    private int num_iters = 5;

    public void OnAIGroupsComplete()
    {
        if (enabled)
        {
            SimulationManager._instance.ResetNPCs();
            StartCoroutine(ProgressImages());
        }
    }

    public IEnumerator ProgressImages()
    {
        yield return StartCoroutine(SampleImages());
        Utils.QuitApplication();
    }

    private void Start()
    {
        FishSpawner spawn = FindObjectOfType<FishSpawner>();

        if (spawn == null)
        {
            StartCoroutine(ProgressImages());
        }
    }

    private IEnumerator SampleImages()
    {
        target_trans = Instantiate(target_prefab, track_camera.transform.position + new Vector3(0, 0, distance), Quaternion.identity).transform;
        target_trans.GetComponent<Animator>().enabled = false;

        var csv = new StringBuilder();

        if (SimulationManager._instance.data_dir == null)
        {
#if UNITY_EDITOR
            data_dir = "..\\image_generation\\sampling\\";
#else
        data_dir = "..\\..\\..\\image_generation\\sampling\\";
#endif
        }

        else
        {
            data_dir = SimulationManager._instance.data_dir;
        }

        Utils.CleanAndCreateDirectories(new Dictionary<string, bool>()
        {
            { data_dir, true }
        });

        image_dir = data_dir + "images\\";

        string graphics_pipeline = GraphicsSettings.defaultRenderPipeline == null ? "built_in\\" : "hdrp\\";
        image_dir += graphics_pipeline;

        foreach (Resolution res in resolutions)
        {
            string res_path = image_dir + res.width.ToString() + "x" + res.height.ToString() + "\\";

            Utils.CleanAndCreateDirectories(new Dictionary<string, bool>()
            {
                { res_path, true }
            });
        }

        int num_images = 10;

        if (SimulationManager._instance.num_images != null)
        {
            try
            {
                num_images = int.Parse(SimulationManager._instance.num_images);
            }
            catch (Exception e)
            {
                Debug.LogError(e.Message);
            }
        }

        float rot_step = 360 / num_images;

        foreach (Resolution res in resolutions)
        {
            string res_path = image_dir + res.width.ToString() + "x" + res.height.ToString() + "\\";

            for (int current_img = 0; current_img < num_images; ++current_img)
            {
                /* Manual calculation */
                //float opposite = Mathf.Sin((degree * Mathf.PI) / 180) * sample_distance;
                //float adjacent = Mathf.Cos((degree * Mathf.PI) / 180) * radius;

                //transform.position = target_obj.transform.position + new Vector3(opposite, 0, -adjacent);
                //transform.Rotate(-Vector3.up);

                track_camera.transform.parent.RotateAround(target_trans.position, Vector3.up, rot_step);

                /* Benchmark 5 times for last image of this res */
                for (int i = current_img == current_img - 1 ? 0 : num_iters; i < num_iters + 1; ++i)
                {
                    float start_time = Time.realtimeSinceStartup;
                    yield return StartCoroutine(Utils.TakeScreenshot(new Tuple<int, int>(res.width, res.height), track_camera, res_path + current_img.ToString() + ".jpg"));
                    float time_taken = Time.realtimeSinceStartup - start_time;

                    /* Time to record time taken */
                    if (current_img == num_images - 1 && i != 0)      // Prioritising caching of the OS
                    {
                        var newLine = $"{res.width.ToString() + "x" + res.height.ToString()},{graphics_pipeline},{time_taken}";
                        csv.AppendLine(newLine);
                    }
                }
            }
        }

        File.AppendAllText(data_dir + "timings.csv", csv.ToString());

        Destroy(target_trans.gameObject);

        yield return null;
    }
}
