public class CoGroupingComparator extends DeserializerComparator < IndexTuple > { public int compare ( byte [ ] b1 , int s1 , int l1 , byte [ ] b2 , int s2 , int l2 ) { try { lhsBuffer . reset ( b1 , s1 , l1 ) ; rhsBuffer . reset ( b2 , s2 , l2 ) ; lhsStream . readVInt ( ) ; rhsStream . readVInt ( ) ; return compareTuples ( null , groupComparators ) ; } catch ( IOException exception ) { throw new CascadingException ( exception ) ; } finally { lhsBuffer . clear ( ) ; rhsBuffer . clear ( ) ; } } public int compare ( IndexTuple lhs , IndexTuple rhs ) { return compareTuples ( groupComparators , lhs . getTuple ( ) , rhs . getTuple ( ) ) ; } }