using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Rendering.PostProcessing;
using static Server;

[RequireComponent(typeof(Rigidbody))][RequireComponent(typeof(FloaterContainer))]
public class ROVController : MonoBehaviour
{
    public Material tracking_material;

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

    public Resolution cam_resolution;

    public Material underwater_skybox_mat;
    public Material ground_skybox_mat;

    public PostProcessVolume volume;

    private ColorGrading m_colour_grading;

    private Color m_color_grading_filter_start;
    private float max_depth = 1000;
    public Transform water_trans;
    private float m_distance_undewater;

    public Camera first_person_cam;

    public float opt_d;

    public float max_d;

    private List<float> radii_to_draw;

    private int num_segments;

    private LineRenderer curr_fwd;

    private LineRenderer optimum_fwd;

    public LayerMask gizmo_mask;

    public void OnActionReceived(JsonMessage param)
    {
        /* For freeze/hybrid, allow physics */
        if (Time.timeScale == 0)
        {
            Time.timeScale = 1;
        }

        StartCoroutine(SendImageData());
    }

    private void Start()
    {
        GameObject target_line = new GameObject();
        target_line.transform.localPosition = Vector3.zero;
        target_line.layer = (int)Mathf.Log(gizmo_mask.value, 2);

        optimum_fwd = target_line.AddComponent<LineRenderer>();
        optimum_fwd.SetPosition(0, transform.position);
        optimum_fwd.material = tracking_material;
        optimum_fwd.material.color = Color.yellow;
        optimum_fwd.startWidth = 0.05f;
        optimum_fwd.endWidth = 0.05f;
        optimum_fwd.positionCount = 2;
        target_line.transform.parent = transform;

        GameObject fwd_line = new GameObject();
        fwd_line.layer = (int)Mathf.Log(gizmo_mask.value, 2);

        curr_fwd = fwd_line.AddComponent<LineRenderer>();
        curr_fwd.SetPosition(0, transform.position);
        curr_fwd.material = tracking_material;
        curr_fwd.material.color = Color.cyan;
        curr_fwd.startWidth = 0.05f;
        curr_fwd.endWidth = 0.05f;
        curr_fwd.positionCount = 2;
        fwd_line.transform.parent = transform;
        fwd_line.transform.localPosition = Vector3.zero;

        num_segments = 100;

        if (SimulationManager._instance.server != null)
        {
            opt_d = SimulationManager._instance.server.json_server_config.payload.serverConfig.envConfig.optD;
            max_d = SimulationManager._instance.server.json_server_config.payload.serverConfig.envConfig.maxD;

            radii_to_draw = new List<float>
            {
                opt_d - max_d,
                opt_d,
                opt_d + max_d
            };
        }

        else
        {
            opt_d = 6;
            max_d = 4;

            radii_to_draw = new List<float>
            {
                opt_d - max_d,
                opt_d,
                opt_d + max_d
            };
        }

        is_underwater = water_trans.position.y > transform.position.y;
        RenderSettings.skybox = is_underwater ? underwater_skybox_mat : ground_skybox_mat;
        volume.profile.TryGetSettings(out m_colour_grading);
        m_color_grading_filter_start = m_colour_grading.colorFilter.value;

        m_rb = GetComponent<Rigidbody>();
        m_rb.drag = air_drag;
        m_rb.angularDrag = angular_air_drag;
        m_rb.useGravity = false;         // We want to optimize 2 actions, disable initial biasing of gravity

        if (SimulationManager._instance.game_state == Enums.E_Game_State.IMAGE_SAMPLING ||
SimulationManager._instance.game_state == Enums.E_Game_State.VAE_GEN || SimulationManager._instance.game_state == Enums.E_Game_State.SCREENSHOT)
        {
            m_rb.isKinematic = true;
        }

        cam_resolution = new Resolution()
        {
            width = 64,
            height = 64
        };

        if (SimulationManager._instance.server != null && SimulationManager._instance.server.json_server_config.msgType.Length > 0)
        {
            cam_resolution = new Resolution()
            {
                width = SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.camConfig.resolution[0],
                height = SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.camConfig.resolution[1]
            };

            m_rb.mass += SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.structureConfig.ballastMass;
        }

        SimulationManager._instance.event_master.rov_initialised_event.Raise(gameObject);
    }

    void Update()
    {
        m_distance_undewater = water_trans.position.y - transform.position.y;
        CheckCameraEffects();
    }

    private void LateUpdate()
    {
        if(SimulationManager._instance.game_state == Enums.E_Game_State.AUTOMATION_TRAINING || SimulationManager._instance.game_state == Enums.E_Game_State.REGULAR)
        {
            foreach (Transform trans in target_transforms)
            {
                int idx = 0;

                foreach (LineRenderer line in trans.GetComponentsInChildren<LineRenderer>())
                {
                    float delta_theta = (float)(2.0 * Mathf.PI) / num_segments;
                    float theta = 0f;

                    for (int i = 0; i < num_segments + 1; i++)
                    {
                        float x = radii_to_draw[idx] * Mathf.Sin(theta);
                        float z = radii_to_draw[idx] * Mathf.Cos(theta);
                        line.SetPosition(i, new Vector3(x, 0, z));
                        theta += delta_theta;
                    }

                    idx++;
                }
            }

            optimum_fwd.SetPosition(0, transform.position);
            optimum_fwd.SetPosition(1, target_transforms[0].position);

            curr_fwd.SetPosition(0, transform.position);
            Vector3 dir = transform.position + transform.forward;
            dir.y = transform.position.y;
            curr_fwd.SetPosition(1, dir);
        }
    }

    private void OnDestroy()
    {
    }

    private void FixedUpdate()
    {
        m_rb.drag = is_underwater ? water_drag : air_drag;
        m_rb.angularDrag = is_underwater ? angular_water_drag : angular_air_drag;
    }

    public void OnAIGroupsComplete()
    {
        if (SimulationManager._instance.server != null && SimulationManager._instance.server.IsConnectionValid() && !SimulationManager._instance.in_manual_mode)
        {
            StartCoroutine(SendImageData());
        }
    }

    public void AddAsTarget(Transform t)
    {
        target_transforms.Add(t);

        for (int i = 0; i < radii_to_draw.Count; ++i)
        {
            GameObject rend = new GameObject();
            rend.layer = (int)Mathf.Log(gizmo_mask.value, 2);

            LineRenderer line = rend.AddComponent<LineRenderer>();
            line.material = tracking_material;
            line.material.color = i == 0 || i == 2 ? Color.red : Color.green;
            line.startWidth = 0.1f;
            line.endWidth = 0.1f;
            line.positionCount = num_segments + 1;
            line.useWorldSpace = false;

            rend.transform.parent = t;
            rend.transform.localPosition = Vector3.zero;
        }
    }

    private IEnumerator SendImageData()
    {
        DirectoryInfo image_dir = SimulationManager._instance.debug_logs ?
       new DirectoryInfo(Path.GetFullPath(Path.Combine(SimulationManager._instance.image_dir.FullName, "episode_" +
       SimulationManager._instance.server.episode_num.ToString() + "_image_" + SimulationManager._instance.server.obsv_num.ToString() + ".jpg"))) : null;

            yield return StartCoroutine(Utils.TakeScreenshot(cam_resolution, first_person_cam, image_dir, byte_image =>
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
