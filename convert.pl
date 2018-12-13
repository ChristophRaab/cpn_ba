#!/usr/bin/perl


# URL that generated this regex:
# http://txt2re.com/index.php3?s=[LVQ%20with%20sigma=5.000000%20and%202%20prototypes%20per%20class;%20using%20batch%20size%201%20on%201%20epochs]%20Correctly%20classified%209%20out%20of%20836%20samples%20(827%20wrong)%20=%3E%20Accuracy%20of%201.076555%&10&57&54&55&59&34&33&9&-285

$re1='^(\\[)';	# Any Single Character 1
$re2='.*?';	# Non-greedy match on filler
$re3='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])';	# Float 1
$re4='.*?';	# Non-greedy match on filler
$re5='(\\d+)';	# Integer Number 1
$re6='.*?';	# Non-greedy match on filler
$re7='(\\d+)';	# Integer Number 2
$re8='.*?';	# Non-greedy match on filler
$re9='(\\d+)';	# Integer Number 3
$re10='.*?';	# Non-greedy match on filler
$re11='(\\d+)';	# Integer Number 4
$re12='.*?';	# Non-greedy match on filler
$re13='(\\d+)';	# Integer Number 5
$re14='.*?';	# Non-greedy match on filler
$re15='(\\d+)';	# Integer Number 6
$re16='.*?';	# Non-greedy match on filler
$re17='([+-]?\\d*\\.\\d+)(?![-+0-9\\.])';	# Float 2

# $txt='[LVQ with sigma=5.000000 and 2 prototypes per class; using batch size 1 on 1 epochs] Correctly classified 9 out of 836 samples (827 wrong) => Accuracy of 1.076555%';
$re=$re1.$re2.$re3.$re4.$re5.$re6.$re7.$re8.$re9.$re10.$re11.$re12.$re13.$re14.$re15.$re16.$re17;
print "accuracy;sigma;prototypes;batchsize;epochs;correct;total;wrong\n";

while(<>) {
	if ($_ =~ m/$re/is)
	{
		$c1=$1;
		$float1=$2;
		$int1=$3;
		$int2=$4;
		$int3=$5;
		$int4=$6;
		$int5=$7;
		$int6=$8;
		$float2=$9;
		print "$float2;$float1;$int1;$int2;$int3;$int4;$int5;$int6\n";
	}
}
