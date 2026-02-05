; ModuleID = 'a.cc'
source_filename = "a.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%class.Animal = type { i32 (...)** }

$_ZZN3Cat5printEvE3cnt = comdat any

@.str = private unnamed_addr constant [25 x i8] c"Dog age = %d, name = %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"dog\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"cat\00", align 1
@.str.3 = private unnamed_addr constant [5 x i8] c"lion\00", align 1
@.str.4 = private unnamed_addr constant [44 x i8] c"\E8\AF\B7\E8\BE\93\E5\85\A5\E5\8A\A8\E7\89\A9\E7\B1\BB\E5\9E\8B\EF\BC\881: \E7\8B\97, 0: \E7\8C\AB\EF\BC\89: \00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.6 = private unnamed_addr constant [18 x i8] c"\E6\89\93\E5\8D\B0\E7\BB\93\E6\9E\9C: %d\0A\00", align 1
@_ZZN3Cat5printEvE3cnt = linkonce_odr dso_local local_unnamed_addr global i32 0, comdat, align 4
@.str.7 = private unnamed_addr constant [35 x i8] c"Cat age = %d, name = %s, cnt = %d\0A\00", align 1
@.str.8 = private unnamed_addr constant [26 x i8] c"Lion age = %d, name = %s\0A\00", align 1
@.str.9 = private unnamed_addr constant [7 x i8] c"global\00", align 1

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @_Z8PrintDog3Dog(i32 %0, i8* %1) local_unnamed_addr #0 {
  %3 = tail call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([25 x i8], [25 x i8]* @.str, i64 0, i64 0), i32 noundef %0, i8* noundef %1)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(i8* nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: mustprogress uwtable
define dso_local void @_Z11PrintAnimalP6Animal(%class.Animal* noundef %0) local_unnamed_addr #2 {
  %2 = bitcast %class.Animal* %0 to void (%class.Animal*)***
  %3 = load void (%class.Animal*)**, void (%class.Animal*)*** %2, align 8, !tbaa !5
  %4 = load void (%class.Animal*)*, void (%class.Animal*)** %3, align 8
  tail call void %4(%class.Animal* noundef nonnull align 8 dereferenceable(8) %0)
  ret void
}

; Function Attrs: uwtable
define dso_local noundef i32 @_Z11PrintAnimalb(i1 noundef zeroext %0) local_unnamed_addr #3 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  br i1 %0, label %2, label %6

2:                                                ; preds = %1
  %3 = tail call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([25 x i8], [25 x i8]* @.str, i64 0, i64 0), i32 noundef 1, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0)) #6
  %4 = tail call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([25 x i8], [25 x i8]* @.str, i64 0, i64 0), i32 noundef 1, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0)) #6
  %5 = tail call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([25 x i8], [25 x i8]* @.str, i64 0, i64 0), i32 noundef 1, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0)) #6
  br label %19

6:                                                ; preds = %1
  %7 = load i32, i32* @_ZZN3Cat5printEvE3cnt, align 4, !tbaa !8
  %8 = add nsw i32 %7, 1
  store i32 %8, i32* @_ZZN3Cat5printEvE3cnt, align 4, !tbaa !8
  %9 = tail call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([35 x i8], [35 x i8]* @.str.7, i64 0, i64 0), i32 noundef 1, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0), i32 noundef %8)
  %10 = tail call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([26 x i8], [26 x i8]* @.str.8, i64 0, i64 0), i32 noundef 1, i8* noundef getelementptr inbounds ([5 x i8], [5 x i8]* @.str.3, i64 0, i64 0))
  %11 = load i32, i32* @_ZZN3Cat5printEvE3cnt, align 4, !tbaa !8
  %12 = add nsw i32 %11, 1
  store i32 %12, i32* @_ZZN3Cat5printEvE3cnt, align 4, !tbaa !8
  %13 = tail call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([35 x i8], [35 x i8]* @.str.7, i64 0, i64 0), i32 noundef 1, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0), i32 noundef %12)
  %14 = tail call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([26 x i8], [26 x i8]* @.str.8, i64 0, i64 0), i32 noundef 1, i8* noundef getelementptr inbounds ([5 x i8], [5 x i8]* @.str.3, i64 0, i64 0))
  %15 = load i32, i32* @_ZZN3Cat5printEvE3cnt, align 4, !tbaa !8
  %16 = add nsw i32 %15, 1
  store i32 %16, i32* @_ZZN3Cat5printEvE3cnt, align 4, !tbaa !8
  %17 = tail call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([35 x i8], [35 x i8]* @.str.7, i64 0, i64 0), i32 noundef 1, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0), i32 noundef %16)
  %18 = tail call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([26 x i8], [26 x i8]* @.str.8, i64 0, i64 0), i32 noundef 1, i8* noundef getelementptr inbounds ([5 x i8], [5 x i8]* @.str.3, i64 0, i64 0))
  br label %19

19:                                               ; preds = %2, %6
  %20 = phi i32 [ 1, %6 ], [ 2, %2 ]
  ret i32 %20
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #4

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  %1 = alloca i8, align 1
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %1) #6
  %2 = tail call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([44 x i8], [44 x i8]* @.str.4, i64 0, i64 0))
  %3 = call i32 (i8*, ...) @__isoc99_scanf(i8* noundef getelementptr inbounds ([3 x i8], [3 x i8]* @.str.5, i64 0, i64 0), i8* noundef nonnull %1)
  %4 = load i8, i8* %1, align 1, !tbaa !11, !range !13
  %5 = icmp ne i8 %4, 0
  %6 = call noundef i32 @_Z11PrintAnimalb(i1 noundef zeroext %5), !range !14
  %7 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([18 x i8], [18 x i8]* @.str.6, i64 0, i64 0), i32 noundef %6)
  %8 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([25 x i8], [25 x i8]* @.str, i64 0, i64 0), i32 noundef 2, i8* noundef getelementptr inbounds ([7 x i8], [7 x i8]* @.str.9, i64 0, i64 0)) #6
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %1) #6
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @__isoc99_scanf(i8* nocapture noundef readonly, ...) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #5 = { mustprogress norecurse uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!5 = !{!6, !6, i64 0}
!6 = !{!"vtable pointer", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !7, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"bool", !10, i64 0}
!13 = !{i8 0, i8 2}
!14 = !{i32 1, i32 3}
