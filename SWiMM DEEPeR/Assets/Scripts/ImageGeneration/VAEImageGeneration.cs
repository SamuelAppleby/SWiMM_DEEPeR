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

    [SerializeField]
    private Resolution resolution;

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

    IEnumerator Start()
    {
        yield return StartCoroutine(GenerateVAEImages());
        Utils.QuitApplication();
    }

    private IEnumerator GenerateVAEImages()
    {
        SimulationManager._instance.ResetNPCs();

        target_trans = Instantiate(target_prefab, Vector3.zero, Quaternion.identity).transform;

        var csv = new StringBuilder();

#if UNITY_EDITOR
        data_dir = "../image_generation/vae/";
#else
        data_dir = "../../../image_generation/vae/";
#endif     

        image_dir = data_dir + "images/";

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

        target_trans.transform.parent = track_camera.transform.parent;     // want the dolphin to be a child of ourselves to avoid recalculating world space

        //int start = Directory.GetFiles(image_dir).Length;
        int start = 499990;

        for (int i = start + 1; i <= start + SimulationManager._instance.num_images; ++i)
        {
            Vector3 new_pos = new Vector3(0, -4, 0);
            new_pos.x = Random.Range(camera_x_range.min, camera_x_range.max);
            new_pos.z = Random.Range(camera_z_range.min, camera_z_range.max);

            track_camera.transform.position = new_pos;

            float rover_yaw = Random.Range(camera_yaw_range.min, camera_yaw_range.max);
            Quaternion rover_yaw_q = Quaternion.Euler(new Vector3(0, rover_yaw, 0));

            track_camera.transform.rotation = rover_yaw_q;

            float new_r = Random.Range(r_range.min, r_range.max);
            float new_theta = Random.Range(theta_range.min, theta_range.max);

            Vector3 dolphin_rel = PolarTranslation(new_r, new_theta);
            //Vector3 world_pos = ConvertTBodyToWorld(dolphin_rel, rover_trans.position, rover_trans.rotation);

            target_trans.localPosition = dolphin_rel;

            float psi_rel = Random.Range(psi_range.min, psi_range.max);
            //float dolphn_yaw = rover_yaw + psi_rel;
            Quaternion dolphin_yaw_q = Quaternion.Euler(new Vector3(0, psi_rel, 0));

            target_trans.localRotation = dolphin_yaw_q;

            yield return StartCoroutine(Utils.TakeScreenshot(new Tuple<int, int>(resolution.width, resolution.height), track_camera, image_dir + "image_" + i.ToString() + ".jpg"));
            var newLine = $"{new_r}, {new_theta * Mathf.Rad2Deg}, {psi_rel}";
            csv.AppendLine(newLine);
        }

        File.AppendAllText(data_dir + "results.csv", csv.ToString());

        Destroy(target_trans.gameObject);
    }
}
