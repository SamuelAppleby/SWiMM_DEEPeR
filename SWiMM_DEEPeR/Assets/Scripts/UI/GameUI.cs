using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using static Enums;
using static Server;

public class GameUI : MonoBehaviour
{
    [SerializeField]
    private FishSpawner spawner;

    [SerializeField]
    private ROVController third_person_movement;

    [SerializeField]
    private ROVControls third_person_controls;

    [SerializeField]
    private TextMeshProUGUI fps_value_text;

    [SerializeField]
    private TextMeshProUGUI episode_num;

    [SerializeField]
    private TextMeshProUGUI observations_sent;

    [SerializeField]
    private TextMeshProUGUI actions_received;

    [SerializeField]
    private TextMeshProUGUI resets_received;

    [SerializeField]
    private TextMeshProUGUI position_value_text;

    [SerializeField]
    private TextMeshProUGUI orienation_value_text;

    [SerializeField]
    private TextMeshProUGUI linear_velocity_value_text;

    [SerializeField]
    private TextMeshProUGUI angular_velocity_value_text;

    [SerializeField]
    private TextMeshProUGUI movement_input_value_text;

    [SerializeField]
    private TextMeshProUGUI rotation_input_value_text;

    [SerializeField]
    private TextMeshProUGUI control_mode_value_text;

    [SerializeField]
    private TextMeshProUGUI movement_value_text;

    [SerializeField]
    private TextMeshProUGUI rotation_value_text;

    [SerializeField]
    private TextMeshProUGUI water_shader_text;

    [SerializeField]
    private TextMeshProUGUI global_lighting_text;

    [SerializeField]
    private TextMeshProUGUI total_ai_text;

    [SerializeField]
    private TextMeshProUGUI far_plane_text;

    private void Start()
    {
        if (SimulationManager._instance.server != null)
        {
            episode_num.text = SimulationManager._instance.server.episode_num.ToString();
            resets_received.text = SimulationManager._instance.server.resets_received.ToString();
        }
    }

    public void OnActionReceived(JsonMessage param)
    {
        actions_received.text = (int.Parse(actions_received.text)+1).ToString();
    }

    public void OnObservationSent()
    {
        observations_sent.text = SimulationManager._instance.server.obsv_num.ToString();
    }

    void FixedUpdate()
    {
        fps_value_text.text = SimulationManager._instance.avgFrameRate.ToString();

        position_value_text.text = third_person_movement.transform.position.ToString();
        orienation_value_text.text = third_person_movement.transform.rotation.eulerAngles.ToString();
        linear_velocity_value_text.text = third_person_movement.m_rb.velocity.ToString();
        angular_velocity_value_text.text = third_person_movement.m_rb.angularVelocity.ToString();
        movement_input_value_text.text = third_person_controls.input_linear.ToString();
        rotation_input_value_text.text = third_person_controls.input_angular.ToString();
        control_mode_value_text.text = third_person_controls.dive_mode.ToString();
        movement_value_text.text = third_person_controls.desired_move.ToString();
        rotation_value_text.text = third_person_controls.desired_rotation.ToString();

        total_ai_text.text = spawner.GetTotalNPCs().ToString();
        far_plane_text.text = third_person_movement.first_person_cam.farClipPlane.ToString();
    }
}
