using Cinemachine;
using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
//using UnityEngine.Rendering.PostProcessing;
using static Server;

[RequireComponent(typeof(Rigidbody))][RequireComponent(typeof(FloaterContainer))]
public class ROVController : MonoBehaviour
{
    [HideInInspector]
    public List<Transform> target_transforms = new List<Transform>();

    public float air_drag;
    public float angular_air_drag;
    public float water_drag;
    public float angular_water_drag;

    [HideInInspector]
    public Rigidbody m_RigidBody;

    public LayerMask water_mask;

    private List<string> collision_objects_list = new List<string>();
    private bool m_is_underwater;

    public Tuple<int, int> resolution;

    public Material underwater_skybox_mat;
    public Material ground_skybox_mat;

    //public PostProcessVolume volume;

    //private ColorGrading m_colour_grading;

    private Color m_color_grading_filter_start;
    private float max_depth = 1000;
    public GameObject water_collider;
    private float top_of_water;
    private float m_distance_undewater;

    private int screenshot_count = 0;

    List<Tuple<int, int>> valid_resolutions = new List<Tuple<int, int>> { new Tuple<int, int>(256, 256), new Tuple<int, int>(512, 512), new Tuple<int, int>(1024, 1024),
    new Tuple<int, int>(2048, 2048)};

    private ROVControls controls;

    public Camera first_person_cam;

    public void OnJsonControls(JsonMessage param)
    {
       StartCoroutine(SendImageData());
    }

    private void Start()
    {
        controls = GetComponent<ROVControls>();
        top_of_water = water_collider.transform.position.y + (water_collider.GetComponent<BoxCollider>().size.y / 2);
        //volume.profile.TryGetSettings(out m_colour_grading);
        //m_color_grading_filter_start = m_colour_grading.colorFilter.value;

        m_RigidBody = GetComponent<Rigidbody>();
        m_RigidBody.drag = air_drag;
        m_RigidBody.angularDrag = angular_air_drag;

        resolution = new Tuple<int, int>(2048,2048);

        if (SimulationManager._instance.server != null && SimulationManager._instance.server.json_server_config.msgType.Length > 0)
        {
            resolution = new Tuple<int, int>(SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.camConfig.resolution[0], SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.camConfig.resolution[1]);
            m_RigidBody.mass += SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.structureConfig.ballastMass;
        }

        EventMaster._instance.rov_initialised_event.Raise(gameObject);
    }

    void Update()
    {
        m_distance_undewater = top_of_water - transform.position.y;
        CheckCameraEffects();
    }

    private void FixedUpdate()
    {
        m_RigidBody.drag = m_is_underwater ? water_drag : air_drag;
        m_RigidBody.angularDrag = m_is_underwater ? angular_water_drag : angular_air_drag;
    }

    private void LateUpdate()
    {
        if (Input.GetKeyUp(KeyCode.F11))
        {
            StartCoroutine(TakeScreenshot(valid_resolutions[screenshot_count], SimulationManager._instance.debug_config.image_dir));
        }
    }

    public void OnAIGroupsComplete()
    {
        StartCoroutine(SendImageData());
    }

    public IEnumerator TakeScreenshot(Tuple<int,int> res, string dir, System.Action<byte[]> callback = null)
    {
        yield return new WaitForEndOfFrame();

        RenderTexture rt = new RenderTexture(res.Item1, res.Item2, 24);
        first_person_cam.targetTexture = rt;
        first_person_cam.Render();
        RenderTexture.active = rt;
        Texture2D screen_shot = new Texture2D(res.Item1, res.Item2, TextureFormat.RGB24, false);
        screen_shot.ReadPixels(new Rect(0, 0, res.Item1, res.Item2), 0, 0);
        screen_shot.Apply();
        first_person_cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
        Destroy(screen_shot);

        if (dir != null)
        {
            if(SimulationManager._instance.server != null)
            {
                File.WriteAllBytes(dir + "image_" + SimulationManager._instance.server.current_obsv_num.ToString() + ".jpg", screen_shot.EncodeToJPG());
            }
            else
            {
                File.WriteAllBytes(dir + "image_" + screenshot_count + ".jpg", screen_shot.EncodeToJPG());
            }
        }

        screenshot_count++;

        if(callback != null)
        {
            callback.Invoke(screen_shot.EncodeToJPG());
        }
    }

    private IEnumerator SendImageData()
    {
        if (SimulationManager._instance.server != null && SimulationManager._instance.server.IsConnectionValid() && !SimulationManager._instance.in_manual_mode)
        {
            yield return StartCoroutine(TakeScreenshot(resolution, SimulationManager._instance.debug_config.image_dir, byte_image =>
            {
                TargetObject[] targetPositions = new TargetObject[target_transforms.Count];
                int pos = 0;

                foreach (Transform trans in target_transforms)
                {
                    targetPositions[pos] = new TargetObject
                    {
                        position = new float[] { trans.position.x, trans.position.y, trans.position.z },
                        fwd = new float[] { trans.forward.x, trans.forward.y, trans.forward.z }
                    };
                    pos++;
                }

                SimulationManager._instance.server.json_str_obsv = JsonConvert.SerializeObject(new DataToSend
                {
                    msg_type = "on_telemetry",
                    payload = new Payload_Data
                    {
                        seq_num = SimulationManager._instance.server.current_packet_num,
                        obsv_num = SimulationManager._instance.server.current_obsv_num,
                        jpg_image = byte_image,
                        position = Utils.Vector3ToFloatArray(transform.position),
                        collision_objects = collision_objects_list.ToArray(),
                        fwd = Utils.Vector3ToFloatArray(transform.forward),
                        targets = targetPositions
                    }
                });
            }));
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        collision_objects_list.Add(collision.gameObject.tag);
    }

    private void OnCollisionStay(Collision collision)
    {
    }

    private void OnCollisionExit(Collision collision)
    {
        /* Physics ticks may happen before message is sent, so persist reset on collion */
        //collision_objects_list.Remove(collision.gameObject.tag);
    }

    private void OnTriggerEnter(Collider other)
    {
        m_is_underwater = ((1 << other.gameObject.layer) & water_mask) != 0;
        controls.object_underwater = m_is_underwater;
    }

    private void OnTriggerStay(Collider other)
    {
        m_is_underwater = ((1 << other.gameObject.layer) & water_mask) != 0;
        controls.object_underwater = m_is_underwater;
    }

    private void OnTriggerExit(Collider other)
    {
        if (m_is_underwater && ((1 << other.gameObject.layer) & water_mask) != 0)
        {
            m_is_underwater = false;
            controls.object_underwater = m_is_underwater;
        }
    }

    public void CheckCameraEffects()
    {
        Collider[] hit_colliders = Physics.OverlapSphere(first_person_cam.transform.position, 0.1f);

        foreach (Collider col in hit_colliders)
        {
            if (((1 << col.gameObject.layer) & water_mask) != 0)
            {
                RenderSettings.skybox = underwater_skybox_mat;

                if(m_distance_undewater > 0)
                {
                    float depth_ratio_underwater_clamped = 1 - Math.Clamp(m_distance_undewater / max_depth, 0, 1);

                    //m_colour_grading.colorFilter.value.r = m_color_grading_filter_start.r * depth_ratio_underwater_clamped;
                    //m_colour_grading.colorFilter.value.g = m_color_grading_filter_start.g * depth_ratio_underwater_clamped;
                    //m_colour_grading.colorFilter.value.b = m_color_grading_filter_start.b * depth_ratio_underwater_clamped;
                }

                return;
            }
        }

        RenderSettings.skybox = ground_skybox_mat;
    }
}
