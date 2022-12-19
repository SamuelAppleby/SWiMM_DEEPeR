using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

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

    void FixedUpdate()
    {
        fps_value_text.text = SimulationManager._instance.avgFrameRate.ToString();

        if (SimulationManager._instance.server != null)
        {
            observations_sent.text = SimulationManager._instance.server.current_obsv_num.ToString();
            actions_received.text = SimulationManager._instance.server.actions_received.ToString();
            resets_received.text = SimulationManager._instance.server.resets_received.ToString();
        }

        position_value_text.text = third_person_movement.transform.position.ToString();
        orienation_value_text.text = third_person_movement.transform.rotation.eulerAngles.ToString();
        linear_velocity_value_text.text = third_person_movement.m_RigidBody.velocity.ToString();
        angular_velocity_value_text.text = third_person_movement.m_RigidBody.angularVelocity.ToString();
        movement_input_value_text.text = third_person_controls.input_linear.ToString();
        rotation_input_value_text.text = third_person_controls.input_angular.ToString();
        control_mode_value_text.text = third_person_controls.dive_mode.ToString();
        movement_value_text.text = third_person_controls.desired_move.ToString();
        rotation_value_text.text = third_person_controls.desired_rotation.ToString();

        if (SimulationManager._instance.water_objs.Length > 0)
        {
            water_shader_text.text = SimulationManager._instance.water_objs[0].activeSelf.ToString();
        }

        if (SimulationManager._instance.lighting_objs.Length > 0)
        {
            global_lighting_text.text = SimulationManager._instance.lighting_objs[0].activeSelf.ToString();
        }

        total_ai_text.text = spawner.GetTotalNPCs().ToString();
        far_plane_text.text = third_person_movement.first_person_cam.farClipPlane.ToString();
    }
}
