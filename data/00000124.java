public class SASLAuthOperationImpl extends SASLBaseOperationImpl implements SASLAuthOperation { private static final byte CMD = 0x21; public SASLAuthOperationImpl(String[] m, String s, Map<String, ?> p, CallbackHandler h, OperationCallback c) { super(CMD, m, EMPTY_BYTES, s, p, h, c); } @Override protected byte[] buildResponse(SaslClient sc) throws SaslException { return sc.hasInitialResponse() ? sc.evaluateChallenge(challenge) : EMPTY_BYTES; } @Override public String toString() { return "SASL auth operation"; } }