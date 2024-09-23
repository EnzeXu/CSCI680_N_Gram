public class EuclideanDistance extends CrossTab { private static final long serialVersionUID = 1L; public EuclideanDistance( Pipe previous ) { this( previous, Fields.size( 3 ), new Fields( "n1", "n2", "euclidean" ) ); } public EuclideanDistance( Pipe previous, Fields argumentFieldSelector, Fields fieldDeclaration ) { super( previous, argumentFieldSelector, new Euclidean(), fieldDeclaration ); } protected static class Euclidean extends CrossTabOperation<Double[]> { private static final long serialVersionUID = 1L; public Euclidean() { super( new Fields( "euclidean" ) ); } public void start( FlowProcess flowProcess, AggregatorCall<Double[]> aggregatorCall ) { aggregatorCall.setContext( new Double[]{0d} ); } public void aggregate( FlowProcess flowProcess, AggregatorCall<Double[]> aggregatorCall ) { TupleEntry entry = aggregatorCall.getArguments(); aggregatorCall.getContext()[ 0 ] += Math.pow( entry.getDouble( 0 ) - entry.getDouble( 1 ), 2 ); } public void complete( FlowProcess flowProcess, AggregatorCall<Double[]> aggregatorCall ) { aggregatorCall.getOutputCollector().add( new Tuple( 1 / ( 1 + aggregatorCall.getContext()[ 0 ] ) ) ); } } }