public class ResponseMessageWrongOrderTest extends ResponseMessageBaseCase { @Before @Override public void setUp() { expectedFlags = new LinkedList<TapResponseFlag>(); ByteBuffer binReqTapMutation = ByteBuffer.allocate(24+16+1+8); binReqTapMutation.put(0, (byte)0x80).put(1, (byte)0x41); binReqTapMutation.put(3, (byte)0x01); binReqTapMutation.put(5, (byte)0x06); binReqTapMutation.put(11, (byte)0x09); binReqTapMutation.put(33, (byte)0x03); binReqTapMutation.put(40, (byte)'a'); binReqTapMutation.put(48, (byte)42); responsebytes = binReqTapMutation.array(); instance = new ResponseMessage(responsebytes); } }