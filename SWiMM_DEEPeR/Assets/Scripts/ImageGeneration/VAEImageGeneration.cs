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

    [SerializeField]
    private Range camera_x_range;

    [SerializeField]
    private Range camera_z_range;

    [SerializeField]
    private Range camera_yaw_range;

    [SerializeField]
    private Range r_range;

    [SerializeField]
    private Range psi_range;

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

        DirectoryInfo data_dir = SimulationManager._instance.data_dir;

        if (data_dir == null)
        {
            DirectoryInfo di = new DirectoryInfo(Path.GetFullPath(Directory.GetCurrentDirectory()));

#if UNITY_EDITOR
            data_dir = new DirectoryInfo(Path.GetFullPath(Path.Combine(di.FullName, "image_generation", "vae")));
#else
            data_dir  = new DirectoryInfo(Path.GetFullPath(Path.Combine(di.Parent.Parent.FullName, "image_generation", "vae")));
#endif     
        }

        foreach (Resolution res in SimulationManager._instance.image_generation_resolutions)
        {
            DirectoryInfo parent_dir = new DirectoryInfo(Path.GetFullPath(Path.Combine(data_dir.FullName, res.ToString())));
            DirectoryInfo image_dir = new DirectoryInfo(Path.GetFullPath(Path.Combine(parent_dir.FullName, "images")));

            Utils.CleanAndCreateDirectories(new Dictionary<DirectoryInfo, bool>()
            {
                { parent_dir, false },
                { image_dir, false },
            });
        }

        float theta = Camera.VerticalToHorizontalFieldOfView(track_camera.fieldOfView, track_camera.sensorSize.x / track_camera.sensorSize.y) * 0.5f * Mathf.Deg2Rad;

        var csv = new StringBuilder();

        for (int i = 0; i < SimulationManager._instance.num_images; ++i)
        {
            Vector3 new_pos = new Vector3(0, -4, 0);
            new_pos.x = Random.Range(camera_x_range.min, camera_x_range.max);
            new_pos.z = Random.Range(camera_z_range.min, camera_z_range.max);

            track_camera.transform.parent.position = new_pos;

            float rover_yaw = Random.Range(camera_yaw_range.min, camera_yaw_range.max);

            track_camera.transform.parent.rotation = Quaternion.Euler(new Vector3(0, rover_yaw, 0));

            float new_r = Random.Range(r_range.min, r_range.max);

            float new_theta = Random.Range(-theta, theta);

            Vector3 dolphin_rel = PolarTranslation(new_r, new_theta);
            //Vector3 world_pos = ConvertTBodyToWorld(dolphin_rel, rover_trans.position, rover_trans.rotation);

            target_trans.localPosition = dolphin_rel;

            float psi_rel = Random.Range(psi_range.min, psi_range.max);
            //float dolphn_yaw = rover_yaw + psi_rel;

            target_trans.localRotation = Quaternion.Euler(new Vector3(0, psi_rel, 0));

            foreach (Resolution res in SimulationManager._instance.image_generation_resolutions)
            {
                DirectoryInfo parent_dir = new DirectoryInfo(Path.GetFullPath(Path.Combine(data_dir.FullName, res.ToString())));
                DirectoryInfo image_dir = new DirectoryInfo(Path.GetFullPath(Path.Combine(parent_dir.FullName, "images")));

                yield return StartCoroutine(Utils.TakeScreenshot(res, track_camera, new DirectoryInfo(Path.GetFullPath(Path.Combine(image_dir.FullName, (Directory.GetFiles(image_dir.FullName).Length + 1).ToString() + ".jpg")))));
            }

            var newLine = $"{new_r}, {new_theta * Mathf.Rad2Deg}, {psi_rel}";
            csv.AppendLine(newLine);
        }

        foreach (Resolution res in SimulationManager._instance.image_generation_resolutions)
        {
            DirectoryInfo parent_dir = new DirectoryInfo(Path.GetFullPath(Path.Combine(data_dir.FullName, res.ToString())));
            File.AppendAllText(Path.GetFullPath(Path.Combine(parent_dir.FullName, "state_data.csv")), csv.ToString());
        }

        Destroy(target_trans.gameObject);
    }
}
