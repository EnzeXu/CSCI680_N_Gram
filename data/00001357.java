public class Merge extends Splice { @ConstructorProperties({"pipes"}) public Merge( Pipe... pipes ) { super( null, pipes ); } @ConstructorProperties({"name", "pipes"}) public Merge( String name, Pipe... pipes ) { super( name, pipes ); } }