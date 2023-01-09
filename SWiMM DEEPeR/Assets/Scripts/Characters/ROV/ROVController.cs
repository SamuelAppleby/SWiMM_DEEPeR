using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering.PostProcessing;
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
    public Rigidbody m_rb;

    public LayerMask water_mask;

    private List<string> collision_objects_list = new List<string>();

    [HideInInspector]
    public bool is_underwater;

    public Tuple<int, int> resolution;

    public Material underwater_skybox_mat;
    public Material ground_skybox_mat;

    public PostProcessVolume volume;

    private ColorGrading m_colour_grading;

    private Color m_color_grading_filter_start;
    private float max_depth = 1000;
    public Transform water_trans;
    private float m_distance_undewater;

    private int manual_screenshot_count = 0;

    Tuple<int, int> manual_screenshot_res = new Tuple<int, int>(1920, 1080);

    public Camera first_person_cam;

    public void OnJsonControls(JsonMessage param)
    {
       StartCoroutine(SendImageData());
    }

    private void Start()
    {
        is_underwater = water_trans.position.y > transform.position.y;
        RenderSettings.skybox = is_underwater ? underwater_skybox_mat : ground_skybox_mat;
        volume.profile.TryGetSettings(out m_colour_grading);
        m_color_grading_filter_start = m_colour_grading.colorFilter.value;

        m_rb = GetComponent<Rigidbody>();
        m_rb.drag = air_drag;
        m_rb.angularDrag = angular_air_drag;

        if (SimulationManager._instance.game_state == Enums.E_Game_State.IMAGE_SAMPLING ||
SimulationManager._instance.game_state == Enums.E_Game_State.VAE_GEN)
        {
            m_rb.isKinematic = true;
            return;
        }

        resolution = new Tuple<int, int>(2048,2048);

        if (SimulationManager._instance.server != null && SimulationManager._instance.server.json_server_config.msgType.Length > 0)
        {
            resolution = new Tuple<int, int>(SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.camConfig.resolution[0], SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.camConfig.resolution[1]);
            m_rb.mass += SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.structureConfig.ballastMass;
        }

        SimulationManager._instance.event_master.rov_initialised_event.Raise(gameObject);
    }

    void Update()
    {
        m_distance_undewater = water_trans.position.y - transform.position.y;
        CheckCameraEffects();

        if (Input.GetKeyUp(KeyCode.F11))
        {
            StartCoroutine(Utils.TakeScreenshot(manual_screenshot_res, first_person_cam, SimulationManager._instance.debug_config.image_dir + "image_" + manual_screenshot_count + ".jpg"));
            manual_screenshot_count++;
        }
    }

    private void LateUpdate()
    {
    }

    private void FixedUpdate()
    {
        m_rb.drag = is_underwater ? water_drag : air_drag;
        m_rb.angularDrag = is_underwater ? angular_water_drag : angular_air_drag;
    }

    public void OnAIGroupsComplete()
    {
        StartCoroutine(SendImageData());
    }

    private IEnumerator SendImageData()
    {
        if (SimulationManager._instance.server != null && SimulationManager._instance.server.IsConnectionValid() && !SimulationManager._instance.in_manual_mode)
        {
            yield return StartCoroutine(Utils.TakeScreenshot(resolution, first_person_cam, SimulationManager._instance.debug_config.image_dir + "episode_" + SimulationManager._instance.server.episode_num.ToString() + "_image_" + SimulationManager._instance.server.obsv_num.ToString() + ".jpg", byte_image =>
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
                    payload = new PayloadDataToSend
                    {
                        telemetry_data = new TelemetryData
                        {
                            jpg_image = byte_image,
                            position = Utils.Vector3ToFloatArray(transform.position),
                            collision_objects = collision_objects_list.ToArray(),
                            fwd = Utils.Vector3ToFloatArray(transform.forward),
                            targets = targetPositions
                        }
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
        is_underwater = ((1 << other.gameObject.layer) & water_mask) != 0;

        if (is_underwater)
        {
            RenderSettings.skybox = underwater_skybox_mat;
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (is_underwater && ((1 << other.gameObject.layer) & water_mask) != 0 && m_distance_undewater < 0)
        {
            is_underwater = false;
            RenderSettings.skybox = ground_skybox_mat;
        }
    }

    public void CheckCameraEffects()
    {
        if (is_underwater)
        {
            float depth_ratio_underwater_clamped = 1 - Math.Clamp(m_distance_undewater / max_depth, 0, 1);

            m_colour_grading.colorFilter.value.r = m_color_grading_filter_start.r * depth_ratio_underwater_clamped;
            m_colour_grading.colorFilter.value.g = m_color_grading_filter_start.g * depth_ratio_underwater_clamped;
            m_colour_grading.colorFilter.value.b = m_color_grading_filter_start.b * depth_ratio_underwater_clamped;
        }
    }
}
