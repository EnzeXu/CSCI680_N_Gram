public class RightJoin extends BaseJoiner { public RightJoin ( ) { } @ ConstructorProperties ( { "fieldDeclaration" } ) public RightJoin ( Fields fieldDeclaration ) { super ( fieldDeclaration ) ; } public Iterator < Tuple > getIterator ( JoinerClosure closure ) { return new JoinIterator ( closure ) ; } public int numJoins ( ) { return -1 ; } public static class JoinIterator extends OuterJoin . JoinIterator { public JoinIterator ( JoinerClosure closure ) { super ( closure ) ; } @ Override protected boolean isOuter ( int i ) { return i == 0 && super . isOuter ( i ) ; } } }