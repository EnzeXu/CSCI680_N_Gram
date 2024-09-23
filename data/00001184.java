public class MinByLocally extends AggregateByLocally { public static class MinPartials implements Functor { private final Fields declaredFields ; public MinPartials ( Fields declaredFields ) { this . declaredFields = declaredFields ; if ( declaredFields . size ( ) != 1 ) throw new IllegalArgumentException ( "declared fields may only have one field , got : " + declaredFields ) ; } @ Override public Fields getDeclaredFields ( ) { return declaredFields ; } @ Override public Tuple aggregate ( FlowProcess flowProcess , TupleEntry args , Tuple context ) { if ( context == null ) return args . getTupleCopy ( ) ; else if ( args . getObject ( 0 ) == null ) return context ; Comparable lhs = ( Comparable ) context . getObject ( 0 ) ; Comparable rhs = ( Comparable ) args . getObject ( 0 ) ; if ( ( lhs == null ) || ( lhs . compareTo ( rhs ) > 0 ) ) context . set ( 0 , rhs ) ; return context ; } @ Override public Tuple complete ( FlowProcess flowProcess , Tuple context ) { return context ; } } @ ConstructorProperties ( { "valueField" , "minField" } ) public MinByLocally ( Fields valueField , Fields minField ) { super ( valueField , new MinPartials ( minField ) ) ; } @ ConstructorProperties ( { "pipe" , "groupingFields" , "valueField" , "minField" } ) public MinByLocally ( Pipe pipe , Fields groupingFields , Fields valueField , Fields minField ) { this ( null , pipe , groupingFields , valueField , minField , 0 ) ; } @ ConstructorProperties ( { "pipe" , "groupingFields" , "valueField" , "minField" , "threshold" } ) public MinByLocally ( Pipe pipe , Fields groupingFields , Fields valueField , Fields minField , int threshold ) { this ( null , pipe , groupingFields , valueField , minField , threshold ) ; } @ ConstructorProperties ( { "name" , "pipe" , "groupingFields" , "valueField" , "minField" } ) public MinByLocally ( String name , Pipe pipe , Fields groupingFields , Fields valueField , Fields minField ) { this ( name , pipe , groupingFields , valueField , minField , USE_DEFAULT_THRESHOLD ) ; } @ ConstructorProperties ( { "name" , "pipe" , "groupingFields" , "valueField" , "minField" , "threshold" } ) public MinByLocally ( String name , Pipe pipe , Fields groupingFields , Fields valueField , Fields minField , int threshold ) { super ( name , pipe , groupingFields , valueField , new MinPartials ( minField ) , threshold ) ; } }