public class Retain extends SubAssembly { @ConstructorProperties({"previous", "retainFields"}) public Retain( Pipe previous, Fields retainFields ) { super( previous ); if( retainFields == null ) throw new IllegalArgumentException( "retainFields may not be null" ); setTails( new Each( previous, retainFields, new Identity(), Fields.RESULTS ) ); } }