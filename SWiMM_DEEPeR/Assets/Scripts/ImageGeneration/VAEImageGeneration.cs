using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using Random = UnityEngine.Random;

public class VAEImageGeneration : MonoBehaviour
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
    private Range camera_x_range;

    [SerializeField]
    private Range camera_z_range;

    [SerializeField]
    private Range camera_yaw_range;

    [SerializeField]
    private Range r_range;

    [HideInInspector]
    private Range theta_range;

    [SerializeField]
    private Range psi_range;

    private float fov_limit;

    private Vector3 PolarTranslation(float r, float theta)
    {
        float corrected = Mathf.PI / 2 + theta;
        return new Vector3(r * Mathf.Cos(corrected), 0, r * Mathf.Sin(corrected));
    }

    private Vector3 ConvertTBodyToWorld(Vector3 relative, Vector3 rover_pos, Quaternion rover_rot)
    {
        Vector3 inversed = Quaternion.Inverse(rover_rot) * relative;
        return rover_pos + inversed;
    }

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
        yield return StartCoroutine(GenerateVAEImages());
        Utils.QuitApplication();
    }

    private void Start()
    {
        FishSpawner spawn = FindObjectOfType<FishSpawner>();

        if(spawn == null)
        {
            StartCoroutine(ProgressImages());
        }
    }

    private IEnumerator GenerateVAEImages()
    {
        target_trans = Instantiate(target_prefab, Vector3.zero, Quaternion.identity).transform;
        target_trans.transform.parent = track_camera.transform.parent;     // want the dolphin to be a child of ourselves to avoid recalculating world space

        var csv = new StringBuilder();

        Resolution res = new Resolution()
        {
            width = 1920,
            height = 1080
        };

        try
        {
            res.width = int.Parse(SimulationManager._instance.vae_resolution.Item1);
            res.height = int.Parse(SimulationManager._instance.vae_resolution.Item2);
        }
        catch (Exception e)
        {
            Debug.LogError(e.Message);
        }

        if (SimulationManager._instance.data_dir == null)
        {
            data_dir = ".." + Path.DirectorySeparatorChar;

#if UNITY_EDITOR
            data_dir += "image_generation" + Path.DirectorySeparatorChar + "vae" + Path.DirectorySeparatorChar;
#else
            data_dir += ".." + Path.DirectorySeparatorChar + ".." + Path.DirectorySeparatorChar + "image_generation" + Path.DirectorySeparatorChar + "vae" + Path.DirectorySeparatorChar;
#endif     
        }

        else
        {
            data_dir = SimulationManager._instance.data_dir;
        }

        data_dir += res.width.ToString() + "x" + res.height.ToString() + Path.DirectorySeparatorChar;

        image_dir = data_dir + "images" + Path.DirectorySeparatorChar;

        Utils.CleanAndCreateDirectories(new Dictionary<string, bool>()
        {
            { data_dir, false },
            { image_dir, false },
        });

        fov_limit = (float)track_camera.fieldOfView * 0.7f;
        fov_limit = fov_limit * 0.5f * Mathf.Deg2Rad;

        theta_range = new Range()
        {
            min = -fov_limit,
            max = fov_limit
        };

        int num_images = 10;

        if(SimulationManager._instance.num_images != null)
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

        int start = Directory.GetFiles(image_dir).Length;

        for (int i = start + 1; i <= start + num_images; ++i)
        {
            Vector3 new_pos = new Vector3(0, -4, 0);
            new_pos.x = Random.Range(camera_x_range.min, camera_x_range.max);
            new_pos.z = Random.Range(camera_z_range.min, camera_z_range.max);

            track_camera.transform.parent.position = new_pos;

            float rover_yaw = Random.Range(camera_yaw_range.min, camera_yaw_range.max);
            Quaternion rover_yaw_q = Quaternion.Euler(new Vector3(0, rover_yaw, 0));

            track_camera.transform.parent.rotation = rover_yaw_q;

            float new_r = Random.Range(r_range.min, r_range.max);
            float new_theta = Random.Range(theta_range.min, theta_range.max);

            Vector3 dolphin_rel = PolarTranslation(new_r, new_theta);
            //Vector3 world_pos = ConvertTBodyToWorld(dolphin_rel, rover_trans.position, rover_trans.rotation);

            target_trans.localPosition = dolphin_rel;

            float psi_rel = Random.Range(psi_range.min, psi_range.max);
            //float dolphn_yaw = rover_yaw + psi_rel;
            Quaternion dolphin_yaw_q = Quaternion.Euler(new Vector3(0, psi_rel, 0));

            target_trans.localRotation = dolphin_yaw_q;

            yield return StartCoroutine(Utils.TakeScreenshot(new Tuple<int, int>(res.width, res.height), track_camera, new DirectoryInfo(image_dir + i.ToString() + ".jpg")));
            var newLine = $"{new_r}, {new_theta * Mathf.Rad2Deg}, {psi_rel}";
            csv.AppendLine(newLine);
        }

        File.AppendAllText(data_dir + "state_data.csv", csv.ToString());

        Destroy(target_trans.gameObject);
    }
}
