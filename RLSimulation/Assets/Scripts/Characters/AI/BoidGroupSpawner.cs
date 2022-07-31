// Copyright (c) 2016 Unity Technologies. MIT license - license_unity.txt
// #NVJOB Simple Boids. MIT license - license_nvjob.txt
// #NVJOB Nicholas Veselov - https://nvjob.github.io
// #NVJOB Simple Boids v1.1.1 - https://nvjob.github.io/unity/nvjob-boids


using System.Collections;
using UnityEngine;

[HelpURL("https://nvjob.github.io/unity/nvjob-boids")]
//[AddComponentMenu("#NVJOB/Boids/Simple Boids")]


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

[System.Serializable]
public class BoidGroup
{
    [Header("General Settings")]
    public Vector2 behavioralCh = new Vector2(2.0f, 6.0f);
    public bool debug;

    [Header("Shoal Settings")]
    [SerializeField]
    [Range(1, 150)] public int numGroups = 5;
    [Range(0, 5000)] public int fragmentedGroup = 50;
    [Range(0, 1)] public float fragmentedFlockYLimit = 0;
    [Range(0, 1.0f)] public float migrationFrequency = 1;
    [Range(0, 1.0f)] public float posChangeFrequency = 0.7f;
    [Range(0, 100)] public float smoothChFrequency = 15f;

    [Header("Fish Settings")]
    [SerializeField]
    public string prefabName;
    [SerializeField]
    [Range(1, 9999)] public int prefabNum = 500;
    [Range(0, 150)] public float fishSpeed = 1;
    [SerializeField]
    [Range(0, 100)] public int fragmentedIndividual = 10;
    [Range(0, 1)] public float fragmentedFishYLimit = 1;
    [Range(0, 10)] public float soaring = 1;
    [Range(0.01f, 500)] public float verticalWawe = 20;
    public bool rotationClamp = false;
    [Range(0, 360)] public float rotationClampValue = 50;
    public Vector2 scaleRandom = new Vector2(1.0f, 1.5f);

    [Header("Danger Settings (one flock)")]
    public bool danger;
    public float dangerRadius = 15;
    public float dangerSpeed = 1.5f;
    public float dangerSoaring = 0.5f;
    public LayerMask dangerLayer;

    public GameObject objectPrefab { get; set; }

    public Transform thisTransform { get; set; }
    public Transform dangerTransform { get; set; }
    public int dangerBird { get; set; }
    public Transform[] fishTransform { get; set; }
    public Transform[] shoalTransform { get; set; }
    public Vector3[] rdTargetPos { get; set; }
    public Vector3[] shoalPos { get; set; }
    public Vector3[] velShoal { get; set; }
    public float[] fishesSpeed { get; set; }
    public float[] fishesSpeedCur { get; set; }
    public float[] spVelocity { get; set; }
    public int[] curentShoal { get; set; }
    public float dangerSpeedCh { get; set; }
    public float dangerSoaringCh { get; set; }
    public float timeTime { get; set; }
    public WaitForSeconds delay0 { get; set; }

    public void Initialise()
    {
        objectPrefab = (GameObject)Resources.Load(prefabName);
    }
}


[System.Serializable]
public class BoidGroupSpawner : MonoBehaviour
{
    [Header("Boid Group Settings")]
    public BoidGroup[] boid_groups;

    public Transform water_surface;

    private void Start()
    {
        if (SimulationManager._instance.server != null && SimulationManager._instance.server.server_config.is_overridden)
        {
            TakeServerOverrides();
        }

        foreach (BoidGroup obj in boid_groups)
        {
            InitialiseGroup(obj);
            obj.thisTransform = transform;

            CreateFlock(obj);
            CreateBird(obj);
            StartCoroutine(BehavioralChange(obj));
            StartCoroutine(Danger(obj));
        }
    }

    public void InitialiseGroup(BoidGroup obj)
    {
        obj.Initialise();
    }

    private void TakeServerOverrides()
    {
        boid_groups = SimulationManager._instance.server.server_config.payload.envConfig.faunaConfig.boidGroups;
    }


    void LateUpdate()
    {
        foreach (BoidGroup obj in boid_groups)
        {
            FlocksMove(obj);
            BirdsMove(obj);
        }
    }

    void FlocksMove(BoidGroup obj)
    {
        for (int f = 0; f < obj.numGroups; f++)
        {
            obj.shoalTransform[f].localPosition = Vector3.SmoothDamp(obj.shoalTransform[f].localPosition, obj.shoalPos[f], ref obj.velShoal[f], obj.smoothChFrequency);
        }
    }

    void BirdsMove(BoidGroup obj)
    {
        float deltaTime = Time.deltaTime;
        obj.timeTime += deltaTime;
        Vector3 translateCur = Vector3.forward * obj.fishSpeed * obj.dangerSpeedCh * deltaTime;
        Vector3 verticalWaweCur = Vector3.up * ((obj.verticalWawe * 0.5f) - Mathf.PingPong(obj.timeTime * 0.5f, obj.verticalWawe));
        float soaringCur = obj.soaring * obj.dangerSoaring * deltaTime;

        for (int b = 0; b < obj.prefabNum; b++)
        {
            if (obj.fishesSpeedCur[b] != obj.fishesSpeed[b]) obj.fishesSpeedCur[b] = Mathf.SmoothDamp(obj.fishesSpeedCur[b], obj.fishesSpeed[b], ref obj.spVelocity[b], 0.5f);
            obj.fishTransform[b].Translate(translateCur * obj.fishesSpeed[b]);

            if (obj.fishTransform[b].position.y > water_surface.position.y)
            {
                obj.fishTransform[b].Translate(new Vector3(0, -50, 0));
            }

            Vector3 tpCh = obj.shoalTransform[obj.curentShoal[b]].position + obj.rdTargetPos[b] + verticalWaweCur - obj.fishTransform[b].position;
            Quaternion rotationCur = Quaternion.LookRotation(Vector3.RotateTowards(obj.fishTransform[b].forward, tpCh, soaringCur, 0));
            if (obj.rotationClamp == false) obj.fishTransform[b].rotation = rotationCur;
            else obj.fishTransform[b].localRotation = BirdsRotationClamp(rotationCur, obj.rotationClampValue);
        }
    }


    IEnumerator Danger(BoidGroup obj)
    {
        if (obj.danger == true)
        {
            obj.delay0 = new WaitForSeconds(1.0f);

            while (true)
            {
                if (Random.value > 0.9f) obj.dangerBird = Random.Range(0, obj.prefabNum);
                obj.dangerTransform.localPosition = obj.fishTransform[obj.dangerBird].localPosition;

                if (Physics.CheckSphere(obj.dangerTransform.position, obj.dangerRadius, obj.dangerLayer))
                {
                    obj.dangerSpeedCh = obj.dangerSpeed;
                    obj.dangerSoaringCh = obj.dangerSoaring;
                    yield return obj.delay0;
                }
                else obj.dangerSpeedCh = obj.dangerSoaringCh = 1;

                yield return obj.delay0;
            }
        }
        else obj.dangerSpeedCh = obj.dangerSoaringCh = 1;
    }

    IEnumerator BehavioralChange(BoidGroup obj)
    {
        while (true)
        {
            yield return new WaitForSeconds(Random.Range(obj.behavioralCh.x, obj.behavioralCh.y));

            //---- Flocks

            for (int f = 0; f < obj.numGroups; f++)
            {
                if (Random.value < obj.posChangeFrequency)
                {
                    Vector3 rdvf = Random.insideUnitSphere * obj.fragmentedGroup;
                    obj.shoalPos[f] = new Vector3(rdvf.x, Mathf.Abs(rdvf.y * obj.fragmentedFlockYLimit), rdvf.z);
                }
            }

            //---- Birds

            for (int b = 0; b < obj.prefabNum; b++)
            {
                obj.fishesSpeed[b] = Random.Range(3.0f, 7.0f);
                Vector3 lpv = Random.insideUnitSphere * obj.fragmentedIndividual;
                obj.rdTargetPos[b] = new Vector3(lpv.x, lpv.y * obj.fragmentedFishYLimit, lpv.z);
                if (Random.value < obj.migrationFrequency) obj.curentShoal[b] = Random.Range(0, obj.numGroups);
            }
        }
    }


    void CreateFlock(BoidGroup obj)
    {
        //--------------

        obj.shoalTransform = new Transform[obj.numGroups];
        obj.shoalPos = new Vector3[obj.numGroups];
        obj.velShoal = new Vector3[obj.numGroups];
        obj.curentShoal = new int[obj.prefabNum];

        for (int f = 0; f < obj.numGroups; f++)
        {
            GameObject nobj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            nobj.SetActive(obj.debug);
            obj.shoalTransform[f] = nobj.transform;
            Vector3 rdvf = Random.onUnitSphere * obj.fragmentedGroup;
            obj.shoalTransform[f].position = obj.thisTransform.position;
            obj.shoalPos[f] = new Vector3(rdvf.x, Mathf.Abs(rdvf.y * obj.fragmentedFlockYLimit), rdvf.z);
            obj.shoalTransform[f].parent = obj.thisTransform;
        }

        //-------------- // For Danger and for flock hunter

        if (obj.danger == true)
        {
            GameObject dobj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            dobj.GetComponent<MeshRenderer>().enabled = obj.debug;
            dobj.layer = gameObject.layer;
            obj.dangerTransform = dobj.transform;
            obj.dangerTransform.parent = obj.thisTransform;
        }
    }

    void CreateBird(BoidGroup obj)
    {
        obj.fishTransform = new Transform[obj.prefabNum];
        obj.fishesSpeed = new float[obj.prefabNum];
        obj.fishesSpeedCur = new float[obj.prefabNum];
        obj.rdTargetPos = new Vector3[obj.prefabNum];
        obj.spVelocity = new float[obj.prefabNum];

        for (int b = 0; b < obj.prefabNum; b++)
        {
            obj.fishTransform[b] = Instantiate(obj.objectPrefab, obj.thisTransform).transform;
            Vector3 lpv = Random.insideUnitSphere * obj.fragmentedIndividual;
            obj.fishTransform[b].localPosition = obj.rdTargetPos[b] = new Vector3(lpv.x, lpv.y * obj.fragmentedFishYLimit, lpv.z);
            obj.fishTransform[b].localScale = Vector3.one * Random.Range(obj.scaleRandom.x, obj.scaleRandom.y);
            obj.fishTransform[b].localRotation = Quaternion.Euler(0, Random.value * 360, 0);
            obj.curentShoal[b] = Random.Range(0, obj.numGroups);
            obj.fishesSpeed[b] = Random.Range(3.0f, 7.0f);
        }
    }

    static Quaternion BirdsRotationClamp(Quaternion rotationCur, float rotationClampValue)
    {
        Vector3 angleClamp = rotationCur.eulerAngles;
        rotationCur.eulerAngles = new Vector3(Mathf.Clamp((angleClamp.x > 180) ? angleClamp.x - 360 : angleClamp.x, -rotationClampValue, rotationClampValue), angleClamp.y, 0);
        return rotationCur;
    }
}