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

    public GameObject objectPrefab { get; set; }

    public AIGroup(string prefab_name, int max_ai, int spawn_amount, bool random_movement, bool randomize_stats)
    {
        prefabName = prefab_name;
        maxAmount = max_ai;
        maxSpawn = spawn_amount;
        randomMovement = random_movement;
        randomizeStats = randomize_stats;
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
}

public class FishSpawner : MonoBehaviour
{
    public Transform[] waypoints;

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
    private Collider m_water_collider;

    [Header("AI Group Settings")]
    public AIGroup[] ai_objects;

    private Vector3 m_spawn_area;

    private bool config_processed = false;

    // Start is called before the first frame update
    void Start()
    {
        //if (SimulationManager._instance.server.IsTcpGood())
        //{
        //    TakeServerOverrides();
        //}

        //CreateSpawnableArea();
        //GetWaypoints();
        //InitialiseGroups();
        //InvokeRepeating("SpawnNPC", 0.5f, spawn_timer);

        if (!SimulationManager._instance.server.IsTcpGood())
        {
            CreateSpawnableArea();
            GetWaypoints();
            InitialiseGroups();
            InvokeRepeating("SpawnNPC", 0.5f, spawn_timer);
        }

    }

    private void Update()
    {
        // TO DO:: When UI scene is integrated (and server config is guranteed) remove below and add above as config will be ready (remove dodgy bool)

        if (SimulationManager._instance.server.IsTcpGood() && !config_processed && SimulationManager._instance.server.server_config.is_overridden)
        {
            TakeServerOverrides();
            CreateSpawnableArea();
            GetWaypoints();
            InitialiseGroups();
            InvokeRepeating("SpawnNPC", 0.5f, spawn_timer);
            config_processed = true;
        }
    }

    private void TakeServerOverrides()
    {
        m_spawn_timer = SimulationManager._instance.server.server_config.payload.envConfig.faunaConfig.spawnTimer;
        m_spawn_container_ratio = SimulationManager._instance.server.server_config.payload.envConfig.faunaConfig.spawnContainerRatio;
        ai_objects = SimulationManager._instance.server.server_config.payload.envConfig.faunaConfig.aiGroups;
    }

    private void CreateSpawnableArea()
    {
        BoxCollider col = m_water_collider as BoxCollider;
        if (col)
        {
            m_spawn_area = new Vector3((m_spawn_container_ratio * 10) * col.size.x, m_spawn_container_ratio * col.size.y, (m_spawn_container_ratio * 10) * col.size.z);
        }
    }

    private void GetWaypoints()
    {
        waypoints = transform.GetComponentsInChildren<Transform>().Where(c => c.gameObject.tag == "Waypoint").ToArray();
    }

    private void InitialiseGroups()
    {
        foreach (AIGroup obj in ai_objects)
        {
            obj.LoadPrefabFromPath();
            if (obj.objectPrefab != null)
            {
                if (obj.randomizeStats)
                {
                    obj.Randomise();
                }

                GameObject m_ai_group_spawn = new GameObject(obj.prefabName);
                m_ai_group_spawn.transform.parent = gameObject.transform;
            }
        }
    }

    private void SpawnNPC()
    {
        foreach (AIGroup obj in ai_objects)
        {
            if (obj.objectPrefab != null && obj.enableSpawner)
            {
                GameObject temp_group = GameObject.Find(obj.prefabName);
                if (temp_group != null && temp_group.GetComponentInChildren<Transform>().childCount < obj.maxAmount)
                {
                    for (int i = 0; i < Random.Range(0, obj.maxSpawn + 1); ++i)
                    {
                        Quaternion random_rotation = Quaternion.Euler(Random.Range(-20, 20), Random.Range(0, 360), 0);
                        Vector3 random_position = GetRandomPosition();
                        if(IsValidLocation(random_position, 30))
                        {
                            GameObject temp_spawn = Instantiate(obj.objectPrefab, random_position, random_rotation);
                            temp_spawn.transform.parent = temp_group.transform;
                            temp_spawn.AddComponent<FishMovement>();
                            temp_spawn.GetComponent<FishMovement>().random_movement = obj.randomMovement;

                            SimulationManager._instance.rover.GetComponent<ThirdPersonMovement>().target_transforms.Add(temp_spawn.transform);
                        }
                    }
                }
            }
        }
    }

    public Vector3 GetRandomPosition()
    {
        Vector3 random_position = new Vector3(Random.Range(-m_spawn_area.x, m_spawn_area.x),
            Random.Range(-m_spawn_area.y, m_spawn_area.y),
            Random.Range(-m_spawn_area.z, m_spawn_area.z)) * .5f;

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
        Gizmos.DrawCube(transform.position, m_spawn_area);
    }
}
