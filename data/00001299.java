public class TapGraph extends TopologyGraph < Tap > { public TapGraph ( Collection < Flow > flows ) { super ( flows . toArray ( new Flow [ flows . size ( ) ] ) ) ; } public TapGraph ( Flow . . . flows ) { super ( flows ) ; } protected Tap getVertex ( Flow flow , Tap tap ) { return tap ; } }