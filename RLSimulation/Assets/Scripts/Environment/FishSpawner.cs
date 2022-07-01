using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.IO;

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
    private int[] rotationOffset;
    [SerializeField]
    public Vector3 rotationOffsetVector;
    [SerializeField]
    public int scale;

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
    [SerializeField]
    private BoxCollider m_water_collider;

    [Header("AI Group Settings")]
    public AIGroup[] ai_groups;

    private Vector3 m_spawn_area;

    private Vector3 m_min_spawn;

    private Vector3 m_max_spawn;

    public static GameObject m_group_parent;

    void Start()
    {
        if (SimulationManager._instance.server.connected && SimulationManager._instance.server.server_config.is_overridden)
        {
            TakeServerOverrides();
        }

        m_group_parent = new GameObject("NPCS");
        m_group_parent.transform.parent = gameObject.transform;
        m_group_parent.transform.position = m_water_collider.transform.position;

        CreateSpawnableArea();
        GetWaypoints();
        InitialiseGroups();
        InvokeRepeating("SpawnNPC", 0.5f, spawn_timer);
    }

    private void TakeServerOverrides()
    {
        m_spawn_timer = SimulationManager._instance.server.server_config.payload.envConfig.faunaConfig.spawnTimer;
        m_spawn_container_ratio = SimulationManager._instance.server.server_config.payload.envConfig.faunaConfig.spawnContainerRatio;
        ai_groups = SimulationManager._instance.server.server_config.payload.envConfig.faunaConfig.aiGroups;

        foreach(AIGroup group in ai_groups)
        {
            group.IntArrayToVector3();
        }
    }

    private void CreateSpawnableArea()
    {
        if (m_water_collider)
        {
            m_spawn_area = m_spawn_container_ratio * new Vector3(m_water_collider.transform.localScale.x * m_water_collider.size.x, m_water_collider.transform.localScale.y 
                * m_water_collider.size.y, m_water_collider.transform.localScale.z * m_water_collider.size.z) / 2;

            m_min_spawn = (m_group_parent.transform.position - m_spawn_area);
            m_max_spawn = (m_group_parent.transform.position + m_spawn_area);
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
        foreach (AIGroup obj in ai_groups)
        {
            if (obj.objectPrefab != null && obj.enableSpawner)
            {
                GameObject temp_group = GameObject.Find("Group: " + obj.prefabName);
                if (temp_group != null && temp_group.GetComponentInChildren<Transform>().childCount < obj.maxAmount)
                {
                    for (int i = 0; i < Random.Range(0, obj.maxSpawn + 1); ++i)
                    {
                        Vector3 random_position = GetRandomPosition();
                        if(IsValidLocation(random_position, 30))
                        {
                            GameObject fixed_rotation = new GameObject(obj.objectPrefab + "fixed rotation");
                            GameObject temp_spawn = Instantiate(obj.objectPrefab, new Vector3(0, 0, 0), Quaternion.Euler(new Vector3(0, 0, 0)));
                            temp_spawn.transform.parent = fixed_rotation.transform;
                            fixed_rotation.transform.parent = temp_group.transform;
                            fixed_rotation.transform.position = random_position;
                            fixed_rotation.transform.rotation = Quaternion.Euler(obj.rotationOffsetVector);
                            fixed_rotation.AddComponent<FishMovement>();
                            fixed_rotation.GetComponent<FishMovement>().random_movement = obj.randomMovement;
                            fixed_rotation.GetComponent<FishMovement>().rotation_offset = fixed_rotation.transform.rotation.eulerAngles;
                            temp_spawn.transform.localScale *= (obj.scale * Random.Range(0.75f, 1.25f));

                            SimulationManager._instance.rover.GetComponent<ThirdPersonMovement>().target_transforms.Add(fixed_rotation.transform);
                        }
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
                if (temp_group != null && temp_group.GetComponentInChildren<Transform>().childCount < obj.maxAmount)
                {
                    total += temp_group.GetComponentInChildren<Transform>().childCount;
                }
            }
        }

        return total;
    }

    public Vector3 GetRandomPosition()
    {
        Vector3 random_position = new Vector3(Random.Range(m_min_spawn.x, m_max_spawn.x),
            Random.Range(m_min_spawn.y, m_max_spawn.y),
            Random.Range(m_min_spawn.z, m_max_spawn.z));

        return random_position;
    }

    public bool IsValidLocation(Vector3 pos, float radius)
    {
        return Physics.CheckSphere(pos, radius);
    }

    public Vector3 RandomWaypoint()
    {
        return waypoints[Random.Range(0, (waypoints.Count() - 1))].position;
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = m_spawn_color;
        Gizmos.DrawCube(m_water_collider.transform.position, m_spawn_area);
    }
}
