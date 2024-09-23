public class FirstNBuffer extends BaseOperation implements Buffer { private final int firstN ; public FirstNBuffer ( ) { super ( Fields . ARGS ) ; firstN = 1 ; } @ ConstructorProperties ( { "firstN" } ) public FirstNBuffer ( int firstN ) { super ( Fields . ARGS ) ; this . firstN = firstN ; } @ ConstructorProperties ( { "fieldDeclaration" } ) public FirstNBuffer ( Fields fieldDeclaration ) { super ( fieldDeclaration . size ( ) , fieldDeclaration ) ; this . firstN = 1 ; } @ ConstructorProperties ( { "fieldDeclaration" , "firstN" } ) public FirstNBuffer ( Fields fieldDeclaration , int firstN ) { super ( fieldDeclaration . size ( ) , fieldDeclaration ) ; this . firstN = firstN ; } @ Property ( name = "firstN" , visibility = Visibility . PUBLIC ) @ PropertyDescription ( "The number of tuples to return . " ) public int getFirstN ( ) { return firstN ; } @ Override public void operate ( FlowProcess flowProcess , BufferCall bufferCall ) { Iterator < TupleEntry > iterator = bufferCall . getArgumentsIterator ( ) ; int count = 0 ; while ( count < firstN && iterator . hasNext ( ) ) { bufferCall . getOutputCollector ( ) . add ( iterator . next ( ) ) ; count++ ; } } }