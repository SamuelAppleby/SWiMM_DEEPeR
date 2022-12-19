using Newtonsoft.Json;
using System;
using System.Dynamic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;
using Task = System.Threading.Tasks.Task;

public class Server
{
    public string last_json_msg = "";
    public int current_packet_num = 0;
    public int current_obsv_num = 0;
    public int actions_received = 0;
    public int resets_received = 0;
    public bool first_observation_sent = false;
    byte[] receive_buffer;

    public TcpClient tcp_client = null;
    public NetworkStream stream = null;

    public UdpClient udp_client = null;


    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct ServerConfig
    {
        public LearningConfig learningConfig;
        public RoverConfig roverConfig;
        public EnvironmentConfig envConfig;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct JsonMessage
    {
        public string msgType;
        public Payload payload;
        public bool is_overriden;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct Payload
    {
        public int seq_num;
        public ServerConfig serverConfig;
        public JsonControls jsonControls;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct ActionSpaceConfig
    {
        public int dimensions;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct LearningConfig
    {
        public ActionSpaceConfig actionSpaceConfig;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct RoverConfig
    {
        public MotorConfig motorConfig;
        public CameraConfig camConfig;
        public StructureConfig structureConfig;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct MotorConfig
    {
        public float stabilityThreshold;
        public float stabilityForce;
        public float[] linearThrustPower;
        public float[] angularThrustPower;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct CameraConfig
    {
        public int[] resolution;
        public int fov;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct StructureConfig
    {
        [Range(0f, 1.2f)]
        public float ballastMass;   // rover spec, 6x200g masses
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct EnvironmentConfig
    {
        public string actionInference;
        public FaunaConfig faunaConfig;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct FaunaConfig
    {
        public float spawnTimer;
        public float spawnContainerRatio;
        public BoidGroup[] boidGroups;
        public AIGroup[] aiGroups;
        public float spawnRadius;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct Buffers
    {
        public int server_send_buffer_size_kb;
        public int client_receive_buffer_size_kb;
        public int server_receive_buffer_size_kb;
        public int client_send_buffer_size_kb;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct DataToSend
    {
        public string msg_type;
        public Payload_Data payload;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct Payload_Data
    {
        public int seq_num;
        public int obsv_num;
        public byte[] jpg_image;
        public float[] position;
        public string[] collision_objects;
        public float[] fwd;
        public TargetObject[] targets;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct TargetObject
    {
        public float[] position;
        public float[] fwd;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct JsonControls
    {
        public float swayThrust;
        public float heaveThrust;
        public float surgeThrust;
        public float pitchThrust;
        public float yawThrust;
        public float rollThrust;
        public float depthHoldMode;
    }

    /* Configs/Messages from server */
    public JsonMessage json_server_config;

    public JsonMessage json_awaiting_training;

    public JsonMessage json_reset_episode;

    public JsonMessage json_rover_controls;

    public JsonMessage json_end_simulation;

    public string json_str_obsv;

    public string json_str_obsv_last;

    IPAddress ipAddress;
    IPEndPoint ipEndPoint;

    public Server()
    {
        json_str_obsv = null;
        json_str_obsv_last = null;
    }

    public bool IsConnectionValid()
    {
        return udp_client != null ? true : stream != null;
    }

    public void Disconnect()
    {
        if(udp_client != null)
        {
            udp_client.Close();
        }
        else
        {
            tcp_client.Close();
        }

        stream = null;
        return;
    }

    public Exception Connect(SimulationManager.NetworkConfig network_config, string ip, int port)
    {
        ipAddress = IPAddress.Parse(ip);
        ipEndPoint = new IPEndPoint(ipAddress, port);

        try
        {
            if (Enums.protocol_mapping[network_config.protocol] == Enums.E_Protocol.UDP)
            {
                udp_client = new UdpClient();
                udp_client.Client.ReceiveBufferSize = network_config.buffers.client_receive_buffer_size_kb * 1024;
                udp_client.Client.SendBufferSize = network_config.buffers.client_send_buffer_size_kb * 1024;
                udp_client.Connect(ipEndPoint);
                return null;
            }
            else
            {
                tcp_client = new TcpClient
                {
                    ReceiveBufferSize = network_config.buffers.client_receive_buffer_size_kb * 1024,
                    SendBufferSize = network_config.buffers.client_send_buffer_size_kb * 1024,
                };

                receive_buffer = new byte[tcp_client.ReceiveBufferSize];
                tcp_client.Connect(ipEndPoint);
                stream = tcp_client.GetStream();
                return null;
            }
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
            return ex;
        }

    }

    public static async Task WaitUntilAsync(Func<bool> predicate, int sleep = 1 / 120)
    {
        while (!predicate())
        {
            await Task.Delay(sleep);
        }
    }

    private bool DataReadyToSend()
    {
        return json_str_obsv != null && json_str_obsv_last == null;
    }

    //public void CastAndOverride<T, P>(JsonMessage<T> data, ref JsonMessage<P> param)
    //{
    //    param = (JsonMessage<P>)(object)(T)(object)data;
    //    param.is_overriden = true;
    //}

    public async Task<GameEvent> ContinueReadWrite() 
    {
        try
        {
            json_str_obsv = JsonConvert.SerializeObject(new DataToSend
            {
                msg_type = "connection_request",
                payload = new Payload_Data
                {
                    seq_num = SimulationManager._instance.server.current_packet_num,
                    obsv_num = SimulationManager._instance.server.current_obsv_num,
                }
            });

            while (IsConnectionValid())
            {
                /* Writing */
                await WaitUntilAsync(DataReadyToSend);

                try
                {
                    if (SimulationManager._instance.debug_config.packets_sent_dir != null)
                    {
                        await File.WriteAllTextAsync(SimulationManager._instance.debug_config.packets_sent_dir + "packet_" + current_packet_num.ToString() + ".json", json_str_obsv);
                    }

                    Debug.Log("Sending: " + json_str_obsv);
                    byte[] _data = Encoding.UTF8.GetBytes(json_str_obsv);

                    if(udp_client != null)
                    {
                        await udp_client.SendAsync(_data, _data.Length);        // Have to send connectionless, don't know who's listening
                    }
                    else
                    {
                        await stream.WriteAsync(_data, 0, _data.Length);
                    }

                    json_str_obsv_last = json_str_obsv;
                    json_str_obsv = null;
                    current_packet_num++;
                }

                catch (Exception e)
                {
                    Debug.LogException(e);
                }

                /* Reading */
                if (udp_client != null)
                {
                    byte[] receiveBytes = udp_client.Receive(ref ipEndPoint);
                    last_json_msg = Encoding.Default.GetString(receiveBytes);
                }
                else
                {
                    int bytes_read = await stream.ReadAsync(receive_buffer, 0, receive_buffer.Length);

                    if (bytes_read == 0)
                    {
                        await Task.Yield();
                        break;
                    }

                    last_json_msg = Encoding.Default.GetString(receive_buffer);
                    Array.Clear(receive_buffer, 0, receive_buffer.Length);
                }

                if (last_json_msg != null)
                {
                    Debug.Log("Received: " + last_json_msg);
                    JsonMessage message = JsonConvert.DeserializeObject<JsonMessage>(last_json_msg);

                    if (SimulationManager._instance.debug_config.packets_received_dir != null)
                    {
                        await File.WriteAllTextAsync(SimulationManager._instance.debug_config.packets_received_dir + "packet_" + message.payload.seq_num + ".json", last_json_msg);
                    }

                    try
                    {
                        switch (message.msgType)
                        {
                            case "process_server_config":
                                json_server_config = message;
                                json_server_config.is_overriden = true;
                                break;
                            case "awaiting_training":
                                json_awaiting_training = message;
                                json_awaiting_training.is_overriden = true;
                                break;
                            case "reset_episode":
                                json_reset_episode = message;
                                json_reset_episode.is_overriden = true;
                                break;
                            case "receive_json_controls":
                                json_rover_controls = message;
                                json_rover_controls.is_overriden = true;
                                break;
                            case "end_simulation":
                                json_end_simulation = message;
                                json_end_simulation.is_overriden = true;
                                break;
                            default:
                                break;
                        }
                    }

                    catch (Exception e)
                    {
                        Debug.LogException(e);
                    }
                }
            }

            /* TCP is no longer good */
            Disconnect();
            return null;
        }
        
        catch(Exception e)
        {
            Debug.LogException(e);
            Disconnect();
            return null;
        }
    }
}
