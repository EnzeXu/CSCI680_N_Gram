public class TapBackfillOperationImpl extends TapOperationImpl implements TapOperation { private final String id ; private final long date ; TapBackfillOperationImpl ( String id , long date , OperationCallback cb ) { super ( cb ) ; this . id = id ; this . date = date ; } @ Override public void initialize ( ) { RequestMessage message = new RequestMessage ( ) ; message . setMagic ( TapMagic . PROTOCOL_BINARY_REQ ) ; message . setOpcode ( TapOpcode . REQUEST ) ; message . setFlags ( TapRequestFlag . BACKFILL ) ; message . setFlags ( TapRequestFlag . SUPPORT_ACK ) ; message . setFlags ( TapRequestFlag . FIX_BYTEORDER ) ; if ( id != null ) { message . setName ( id ) ; } else { message . setName ( UUID . randomUUID ( ) . toString ( ) ) ; } message . setBackfill ( date ) ; setBuffer ( message . getBytes ( ) ) ; } @ Override public void streamClosed ( OperationState state ) { transitionState ( state ) ; } @ Override public String toString ( ) { return "Cmd : tap dump Flags : backfill , ack" ; } }