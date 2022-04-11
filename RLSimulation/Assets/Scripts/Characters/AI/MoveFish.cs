using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveFish : MonoBehaviour
{
    private Rigidbody m_RigidBody;
    private float f = 100;
    //private float fr = 3;
    //private float prevSign = 0;
    System.Random  rnd2;

    // Start is called before the first frame update
    void Start()
    {
        ///rnd1 = new System.Random();
        rnd2 = new System.Random();
        //Debug.Log("x=(" + minX + "," + maxX + ") y=("+minY+","+maxY+") z=("+minZ+","+maxZ+")");
        m_RigidBody = GetComponent<Rigidbody>();
        m_RigidBody.freezeRotation = false;
    }
    
    enum Direction
    {
        L, R, U, D, B, F
    }

    Vector3 getDirection(Direction x)
    {
        switch (x)
        {
            case Direction.L:
                return left;
            case Direction.R:
                return right;
            case Direction.U:
                return up;
            case Direction.D:
                return down;
            case Direction.B:
                return new Vector3(0,0,1);
            case Direction.F:
                return new Vector3(0, 0, -1);
            default:
                return new Vector3(0, 0, 0);
        }
    }

    Direction majDirection(Vector3 vec)
    {
        float x = vec.x;
        float y = vec.y;
        float z = vec.z;
        if (Mathf.Abs(x) < Mathf.Abs(vec.y))
        {
            if (Mathf.Abs(y) < Mathf.Abs(vec.z))
            {
                return y >= 0 ? Direction.U : Direction.D;
            }
            else
            {
                return z >= 0 ? Direction.R : Direction.L;
            }
        } else
        {
            if (Mathf.Abs(x) < Mathf.Abs(vec.z))
            {
                return z >= 0 ? Direction.R : Direction.L;
            }
            else
            {
                return x >= 0 ? Direction.B : Direction.F;
            }
        }
    }

    float projectDirection(Vector3 vec, Direction x)
    {
        switch (x)
        {
            case Direction.L:
            case Direction.R:
                return vec.z;
            case Direction.U:
            case Direction.D:
                return vec.y;
            case Direction.B:
            case Direction.F:
                return vec.x;
            default:
                return 0;
        }
    }



    class Action
    {
        public Vector3 direction;
        public float time;

        public Action(Vector3 dir, float t = 0) {
            direction = dir;
            time = t;
        }

        public Action()
        {
            direction = new Vector3(0, 0, 0);
            time = 0;
        }

        public Quaternion rotate(Vector3 eulerAngles, Action nextAction) {
            return Quaternion.Euler(eulerAngles + Quaternion.FromToRotation(direction, nextAction.direction).eulerAngles);
        }
    }



    Action prevAction = null;
    Action currAction = null;

    static Vector3 left = new Vector3(0, 0, -1);
    static Vector3 right = new Vector3(0, 0, 1);
    static Vector3 back = new Vector3(1, 0, 0);
    static Vector3 forward = new Vector3(-1, 0, 0);
    static Vector3 up = new Vector3(0, 1, 0);
    static Vector3 down = new Vector3(0, -1, 0);

    Action generateNewAction()

    {

        
        
        int direction = rnd2.Next(0, 6);  // creates a number
        float sign = ((direction % 2) == 1) ? 1 : -1; // Generates the "sign" of the direction
        direction = direction / 3;                    // performs the movement along one of the three axes

        //m_RigidBody.freezeRotation = false;
        if (direction == 0)
        {
            //Debug.Log(sign == 1 ? "L" : "R");
            return new Action(sign * left * Time.deltaTime * f);
        }
        else if (direction == 1)
        {
            //Debug.Log(sign == 1 ? "Up" : "Down");
            return new Action(sign * up * Time.deltaTime * f);
        }
        else //if (direction == 2)
        {
            //Debug.Log(sign == 1 ? "FW" : "RW");
            return new Action(sign * forward * Time.deltaTime * f);
        }
    }

    void performAction(Vector3 x, bool doRotate = false, Direction dir = Direction.R)
    {
        //m_RigidBody.freezeRotation = true;
        m_RigidBody.velocity = x * 3;
    }

    const float MAXTIME = 3.0f;


    void handleNewAction(Action newAction)
    {
        prevAction = currAction;
        currAction = newAction;
        //fromAngle = transform.rotation;
        //toAngle = prevAction.rotate(transform.eulerAngles, currAction);
       /// Debug.Log(fromAngle.eulerAngles.ToString()+" --> " +toAngle.eulerAngles.ToString());
        currAction.time = 0;
        currAction.time += Time.deltaTime;
        ///transform.rotation = (Quaternion.Lerp(fromAngle, toAngle, currAction.time / MAXTIME));
        performAction(currAction.direction);
    }

    // Update is called once per frame
    void Update()
    {
        if ((prevAction == null) && (currAction == null))
        {
            currAction = new Action();
            handleNewAction(generateNewAction());
        } else {
            Vector3 changeDir = new Vector3(0, 0, 0);
            bool hasChangeDir = false;
            if ((m_RigidBody.position.y + 2.0) >= 12)
            {
                hasChangeDir = true;
                changeDir += down;
            }
            /*if ((m_RigidBody.position.y + 1.0) <= minY)
            {
                hasChangeDir = true;
                changeDir += up;
            }
            if ((m_RigidBody.position.x + 1.0) <= minX)
            {
                hasChangeDir = true;
                changeDir += back;
            }
            if ((m_RigidBody.position.x + 1.0) <= maxX)
            {
                hasChangeDir = true;
                changeDir += forward;
            }
            if ((m_RigidBody.position.z + 1.0) <= minZ)
            {
                hasChangeDir = true;
                changeDir += right;
            }
            if ((m_RigidBody.position.z - 1.0) <= maxZ)
            {
                hasChangeDir = true;
                changeDir += left;
            }
            */
            if (hasChangeDir)
            {
                handleNewAction(new Action(changeDir));
                return;
            }
           // Debug.Log(currAction.time);
            currAction.time += Time.deltaTime;
            if (currAction.time < MAXTIME) {
                performAction(currAction.direction);
                /// transform.rotation = (Quaternion.Lerp(fromAngle, toAngle, currAction.time / MAXTIME));
            } else {
                handleNewAction(generateNewAction());
            }
        }
    }



    void OnCollisionEnter(Collision col)
    {
        m_RigidBody.freezeRotation = true;
        ///float bounce = 6f; //amount of force to apply
        handleNewAction(new Action(col.contacts[0].normal * 3));
        //performAction(col.contacts[0].normal * 3);
    }
}
