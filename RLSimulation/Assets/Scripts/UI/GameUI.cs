using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class GameUI : MonoBehaviour
{
    [SerializeField]
    private FishSpawner spawner;

    [SerializeField]
    private ThirdPersonMovement third_person_movement;

    [SerializeField]
    private TextMeshProUGUI fps_value_text;

    [SerializeField]
    private TextMeshProUGUI packets_value_text;

    [SerializeField]
    private TextMeshProUGUI depth_hold_mode_text;

    [SerializeField]
    private TextMeshProUGUI movement_input_value_text;

    [SerializeField]
    private TextMeshProUGUI rotation_input_value_text;

    [SerializeField]
    private TextMeshProUGUI movement_value_text;

    [SerializeField]
    private TextMeshProUGUI rotation_value_text;

    [SerializeField]
    private TextMeshProUGUI position_value_text;

    [SerializeField]
    private TextMeshProUGUI orienation_value_text;

    [SerializeField]
    private TextMeshProUGUI water_shader_text;

    [SerializeField]
    private TextMeshProUGUI global_lighting_text;

    [SerializeField]
    private TextMeshProUGUI total_ai_text;

    [SerializeField]
    private TextMeshProUGUI far_plane_text;

    void Update()
    {
        fps_value_text.text = SimulationManager._instance.avgFrameRate.ToString();
        packets_value_text.text = (SimulationManager._instance.server.sequence_num - 1).ToString();
        depth_hold_mode_text.text = third_person_movement.m_depth_hold_mode.ToString();
        movement_input_value_text.text = third_person_movement.movement_controls.movementInputs.ToString();
        rotation_input_value_text.text = third_person_movement.movement_controls.rotationInputs.ToString();
        movement_value_text.text = third_person_movement.desiredMove.ToString();
        rotation_value_text.text = third_person_movement.desiredRotation.ToString();
        position_value_text.text = third_person_movement.transform.position.ToString();
        orienation_value_text.text = third_person_movement.transform.rotation.eulerAngles.ToString();

        if (SimulationManager._instance.water_objs.Length > 0)
        {
            water_shader_text.text = SimulationManager._instance.water_objs[0].activeSelf.ToString();
        }

        if (SimulationManager._instance.lighting_objs.Length > 0)
        {
            global_lighting_text.text = SimulationManager._instance.lighting_objs[0].activeSelf.ToString();
        }

        total_ai_text.text = spawner.GetTotalNPCs().ToString();
        far_plane_text.text = third_person_movement.firstPersonCam.farClipPlane.ToString();
    }
}
