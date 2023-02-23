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

    [SerializeField]
    private float distance;

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
        DirectoryInfo data_dir = SimulationManager._instance.data_dir;

        if (data_dir == null)
        {
            DirectoryInfo di = new DirectoryInfo(Path.GetFullPath(Directory.GetCurrentDirectory()));

#if UNITY_EDITOR
            data_dir  = new DirectoryInfo(Path.GetFullPath(Path.Combine(di.FullName, "image_generation", "sampling")));
#else
            data_dir  = new DirectoryInfo(Path.GetFullPath(Path.Combine(di.Parent.Parent.FullName, "image_generation", "sampling")));
#endif
        }

        Utils.CleanAndCreateDirectories(new Dictionary<DirectoryInfo, bool>()
        {
            { data_dir, true }
        });

        string graphics_pipeline = GraphicsSettings.defaultRenderPipeline == null ? "built_in" : "hdrp";
        DirectoryInfo image_dir = new DirectoryInfo(Path.GetFullPath(Path.Combine(data_dir.FullName, "images", graphics_pipeline)));

        foreach (Resolution res in SimulationManager._instance.image_generation_resolutions)
        {
            Utils.CleanAndCreateDirectories(new Dictionary<DirectoryInfo, bool>()
            {
                { new DirectoryInfo(Path.GetFullPath(Path.Combine(image_dir.FullName, res.ToString()))), true }
            });
        }

        float rot_step = 360 / SimulationManager._instance.num_images;

        foreach (Resolution res in SimulationManager._instance.image_generation_resolutions)
        {
            for (int current_img = 0; current_img < SimulationManager._instance.num_images; ++current_img)
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
                    yield return StartCoroutine(Utils.TakeScreenshot(res, track_camera, 
                        new DirectoryInfo(Path.GetFullPath(Path.Combine(image_dir.FullName, res.ToString(), current_img.ToString() + ".jpg")))));
                    float time_taken = Time.realtimeSinceStartup - start_time;

                    /* Time to record time taken */
                    if (current_img == SimulationManager._instance.num_images - 1 && i != 0)      // Prioritising caching of the OS
                    {
                        var newLine = $"{res},{graphics_pipeline},{time_taken}";
                        csv.AppendLine(newLine);
                    }
                }
            }
        }

        File.AppendAllText(Path.GetFullPath(Path.Combine(data_dir.FullName, "timings.csv")), csv.ToString());

        Destroy(target_trans.gameObject);

        yield return null;
    }
}
