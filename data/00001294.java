public class Cascades { public static Map < String , Tap > tapsMap ( String name , Tap tap ) { return tapsMap ( new String [ ] { name } , Tap . taps ( tap ) ) ; } public static Map < String , Tap > tapsMap ( String [ ] names , Tap [ ] taps ) { Map < String , Tap > map = new HashMap < String , Tap > ( ) ; for ( int i = 0 ; i < names . length ; i++ ) map . put ( names [ i ] , taps [ i ] ) ; return map ; } public static Map < String , Tap > tapsMap ( Pipe pipe , Tap tap ) { return tapsMap ( Pipe . pipes ( pipe ) , Tap . taps ( tap ) ) ; } public static Map < String , Tap > tapsMap ( Pipe [ ] pipes , Tap [ ] taps ) { Map < String , Tap > map = new HashMap < String , Tap > ( ) ; for ( int i = 0 ; i < pipes . length ; i++ ) map . put ( pipes [ i ] . getName ( ) , taps [ i ] ) ; return map ; } public static FlowGraph getFlowGraphFrom ( Cascade cascade ) { return ( ( BaseCascade ) cascade ) . getFlowGraph ( ) ; } public static IdentifierGraph getTapGraphFrom ( Cascade cascade ) { return ( ( BaseCascade ) cascade ) . getIdentifierGraph ( ) ; } }