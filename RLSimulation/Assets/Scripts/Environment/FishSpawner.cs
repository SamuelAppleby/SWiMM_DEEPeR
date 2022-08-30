using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.IO;
using System;
using Random = UnityEngine.Random;

[System.Serializable]
public class AIGroup
{
    [Header("AI Group Stats")]
    [SerializeField]
    public string prefabName;
    [SerializeField]
    [Range(0f, 10000f)]
    public int maxAmount;
    [SerializeField]
    [Range(0f, 1000f)]
    public int maxSpawn;
    [SerializeField]
    public bool enableSpawner;
    [SerializeField]
    public bool randomMovement;
    [SerializeField]
    public bool randomizeStats;
    [SerializeField]
    [HideInInspector]
    public bool[] movementDirections;
    [SerializeField]
    public Vector3 movementDirectionsVector;
    [SerializeField]
    [HideInInspector]
    public int[] rotationOffset;
    [SerializeField]
    public Vector3 rotationOffsetVector;
    [SerializeField]
    public int scale;
    [SerializeField]
    public int minSpeed = 1;
    [SerializeField]
    public int maxSpeed = 7;
    [SerializeField]
    public bool spawnInfront = false;
    public GameObject objectPrefab { get; set; }

    public AIGroup(string prefab_name, int max_ai, int spawn_amount, bool random_movement, bool randomize_stats, Vector3 rotation_offfset, int scale_value)
    {
        prefabName = prefab_name;
        maxAmount = max_ai;
        maxSpawn = spawn_amount;
        randomMovement = random_movement;
        randomizeStats = randomize_stats;
        rotationOffsetVector = rotation_offfset;
        scale = scale_value;
    }

    public void Randomise()
    {
        maxAmount = Random.Range(1, 10000);
        maxSpawn = Random.Range(1, 1000);
    }

    public void LoadPrefabFromPath()
    {
        objectPrefab = (GameObject)Resources.Load(prefabName);
    }

    public void IntArrayToVector3()
    {
        rotationOffsetVector = new Vector3(rotationOffset[0], rotationOffset[1], rotationOffset[2]);
        movementDirectionsVector = new Vector3(Convert.ToInt32(movementDirections[0]), Convert.ToInt32(movementDirections[1]), Convert.ToInt32(movementDirections[2]));
    }
}

public class FishSpawner : MonoBehaviour
{
    private Transform[] waypoints;

    public float spawn_timer { get { return m_spawn_timer; } }
    public float spawn_container_ratio { get { return m_spawn_container_ratio; } }

    [Header("Global Stats")]
    [Range(0f, 600f)]
    [SerializeField]
    private float m_spawn_timer;
    [SerializeField]
    [Range(0f, 1f)]
    private float m_spawn_container_ratio;
    [SerializeField]
    private Color m_spawn_color = new Color(1f, 0f, 0f, 0.3f);

    public float spawn_radius;

    public Transform environment_transform;

    public static GameObject m_group_parent;

    public GameObject water_surface;

    private Vector3 spawn_centre;

    [Header("AI Group Settings")]
    public AIGroup[] ai_groups;

    void Start()
    {
        if (SimulationManager._instance.server != null && SimulationManager._instance.server.server_config.is_overridden)
        {
            TakeServerOverrides();
        }

        m_group_parent = new GameObject("NPCS");
        m_group_parent.transform.parent = environment_transform;

        GetWaypoints();
        InitialiseGroups();
        InvokeRepeating("SpawnNPC", 0.5f, spawn_timer);
    }

    private void TakeServerOverrides()
    {
        m_spawn_timer = SimulationManager._instance.server.server_config.payload.envConfig.faunaConfig.spawnTimer;
        m_spawn_container_ratio = SimulationManager._instance.server.server_config.payload.envConfig.faunaConfig.spawnContainerRatio;
        spawn_radius = SimulationManager._instance.server.server_config.payload.envConfig.faunaConfig.spawnRadius;
        ai_groups = SimulationManager._instance.server.server_config.payload.envConfig.faunaConfig.aiGroups;

        foreach (AIGroup group in ai_groups)
        {
            group.IntArrayToVector3();
        }
    }

    private void GetWaypoints()
    {
        waypoints = transform.GetComponentsInChildren<Transform>().Where(c => c.gameObject.tag == "Waypoint").ToArray();
    }

    private void InitialiseGroups()
    {
        foreach (AIGroup obj in ai_groups)
        {
            obj.LoadPrefabFromPath();
            if (obj.objectPrefab != null)
            {
                if (obj.randomizeStats)
                {
                    obj.Randomise();
                }

                GameObject m_ai_group_spawn = new GameObject("Group: " + obj.prefabName);
                m_ai_group_spawn.transform.parent = m_group_parent.transform;
            }
        }
    }

    private void SpawnNPC()
    {
        foreach (AIGroup group in ai_groups)
        {
            if (group.objectPrefab != null && group.enableSpawner)
            {
                GameObject temp_group = GameObject.Find("Group: " + group.prefabName);
                if (temp_group != null && temp_group.GetComponentInChildren<Transform>().childCount < group.maxAmount)
                {
                    for (int i = 0; i < Random.Range(1, group.maxSpawn + 1); ++i)
                    {
                        GameObject fixed_rotation = new GameObject(group.objectPrefab + "fixed rotation");
                        GameObject temp_spawn = Instantiate(group.objectPrefab, new Vector3(0, 0, 0), Quaternion.Euler(new Vector3(0, 0, 0)));
                        temp_spawn.transform.parent = fixed_rotation.transform;
                        fixed_rotation.transform.parent = temp_group.transform;
                        fixed_rotation.transform.position = group.spawnInfront ? SimulationManager._instance.rover.transform.position +
                            (SimulationManager._instance.rover.transform.forward * 20)
                            : GetRandomValidPosition(group.movementDirectionsVector);
                        fixed_rotation.transform.rotation = Quaternion.Euler(group.rotationOffsetVector);
                        fixed_rotation.AddComponent<FishMovement>();
                        fixed_rotation.GetComponent<FishMovement>().random_movement = group.randomMovement;
                        fixed_rotation.GetComponent<FishMovement>().ai_manager = this;
                        fixed_rotation.GetComponent<FishMovement>().rotation_offset = fixed_rotation.transform.rotation.eulerAngles;
                        fixed_rotation.GetComponent<FishMovement>().m_mix_max_speed = new System.Tuple<float, float>(group.minSpeed, group.maxSpeed);
                        fixed_rotation.GetComponent<FishMovement>().valid_movements = group.movementDirectionsVector;
                        temp_spawn.transform.localScale *= (group.scale * Random.Range(0.75f, 1.25f));

                        SimulationManager._instance.rover.GetComponent<ThirdPersonMovement>().target_transforms.Add(fixed_rotation.transform);
                    }
                }
            }
        }
    }

    public int GetTotalNPCs()
    {
        int total = 0;

        foreach (AIGroup obj in ai_groups)
        {
            if (obj.objectPrefab != null && obj.enableSpawner)
            {
                GameObject temp_group = GameObject.Find("Group: " + obj.prefabName);
                if (temp_group != null)
                {
                    total += temp_group.GetComponentInChildren<Transform>().childCount;
                }
            }
        }

        return total;
    }

    public Vector3 GetRandomValidPosition(Vector3 valid_directions)
    {
        spawn_centre = SimulationManager._instance.rover.transform.position;

        if (SimulationManager._instance.rover.transform.position.y + spawn_radius > water_surface.transform.position.y)
        {
            spawn_centre.y -= (SimulationManager._instance.rover.transform.position.y + spawn_radius - water_surface.transform.position.y);
        }

        Vector3 random_directions = spawn_centre;

        if (valid_directions.x > 0)
        {
            random_directions.x += Random.Range(0, spawn_radius);
        }
        if (valid_directions.y > 0)
        {
            random_directions.y += Random.Range(0, spawn_radius);
        }
        if (valid_directions.z > 0)
        {
            random_directions.z += Random.Range(0, spawn_radius);
        }

        return random_directions;
    }

    public Vector3 RandomWaypoint()
    {
        return waypoints[Random.Range(0, (waypoints.Count() - 1))].position;
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = m_spawn_color;
        Gizmos.DrawSphere(spawn_centre, spawn_radius);
    }
}
