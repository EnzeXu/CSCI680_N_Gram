public class RegexGenerator extends RegexOperation < Pair < Matcher , TupleEntry > > implements Function < Pair < Matcher , TupleEntry > > { @ ConstructorProperties ( { "patternString" } ) public RegexGenerator ( String patternString ) { super ( 1 , Fields . size ( 1 ) , patternString ) ; } @ ConstructorProperties ( { "fieldDeclaration" , "patternString" } ) public RegexGenerator ( Fields fieldDeclaration , String patternString ) { super ( 1 , fieldDeclaration , patternString ) ; if ( fieldDeclaration . size ( ) != 1 ) throw new IllegalArgumentException ( "fieldDeclaration may only declare one field , was " + fieldDeclaration . print ( ) ) ; } @ Override public void prepare ( FlowProcess flowProcess , OperationCall < Pair < Matcher , TupleEntry > > operationCall ) { TupleEntry tupleEntry = new TupleEntry ( operationCall . getDeclaredFields ( ) , Tuple . size ( 1 ) ) ; operationCall . setContext ( new Pair < > ( getPattern ( ) . matcher ( "" ) , tupleEntry ) ) ; } @ Override public void operate ( FlowProcess flowProcess , FunctionCall < Pair < Matcher , TupleEntry > > functionCall ) { String value = functionCall . getArguments ( ) . getString ( 0 ) ; if ( value == null ) value = "" ; Matcher matcher = functionCall . getContext ( ) . getLhs ( ) . reset ( value ) ; while ( matcher . find ( ) ) { functionCall . getContext ( ) . getRhs ( ) . setString ( 0 , matcher . group ( ) ) ; functionCall . getOutputCollector ( ) . add ( functionCall . getContext ( ) . getRhs ( ) ) ; } } }