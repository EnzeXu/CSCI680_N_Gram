public class ReverseGroupingSortingComparator extends GroupingSortingComparator { @ Override public int compare ( byte [ ] b1 , int s1 , int l1 , byte [ ] b2 , int s2 , int l2 ) { return -1 * super . compare ( b1 , s1 , l1 , b2 , s2 , l2 ) ; } @ Override public int compare ( TuplePair lhs , TuplePair rhs ) { return super . compare ( rhs , lhs ) ; } }