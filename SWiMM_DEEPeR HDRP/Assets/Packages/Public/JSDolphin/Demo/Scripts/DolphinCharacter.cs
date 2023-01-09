using UnityEngine;
using System.Collections;

public class DolphinCharacter : MonoBehaviour
{
    Animator dolphinAnimator;
    Rigidbody dolphinRigid;

    public bool isLived = true;

    public float forwardSpeed = 1f;
    public float turnSpeed = .3f;
    public float upDownSpeed = 0f;
    public float rollSpeed = 0f;

    public float maxForwardSpeed = 1f;
    public float maxTurnSpeed = 100f;
    public float maxUpDownSpeed = 100f;
    public float maxRollSpeed = 100f;

    void Start()
    {
        dolphinAnimator = GetComponent<Animator>();
        dolphinRigid = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        Move();
    }

    public void Hit()
    {
        dolphinAnimator.SetTrigger("Hit");
    }

    public void Bite()
    {
        dolphinAnimator.SetTrigger("Bite");
    }

    public void Death()
    {
        dolphinAnimator.SetTrigger("Death");
        isLived = false;
    }

    public void Rebirth()
    {
        dolphinAnimator.SetTrigger("Rebirth");
        isLived = true;
    }

    public void Move()
    {
        if (isLived)
        {
            dolphinRigid.velocity = transform.forward * forwardSpeed * maxForwardSpeed;
            transform.RotateAround(transform.position, -transform.right, upDownSpeed * Time.deltaTime * maxUpDownSpeed);
            transform.RotateAround(transform.position, transform.up, turnSpeed * Time.deltaTime * maxTurnSpeed);
            transform.RotateAround(transform.position, transform.forward, rollSpeed * Time.deltaTime * maxRollSpeed);
            
            dolphinAnimator.SetFloat("Forward", forwardSpeed);
            dolphinAnimator.SetFloat("Turn", turnSpeed);
            dolphinAnimator.SetFloat("UpDown", upDownSpeed);
            dolphinAnimator.SetFloat("Roll", rollSpeed);
        }
    }
}
