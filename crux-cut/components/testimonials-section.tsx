import { Card, CardContent, CardFooter } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

export function TestimonialsSection() {
  return (
    <section id="testimonials" className="py-16 md:py-24 bg-black">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">클라이머들의 후기</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">CruxCut을 사용한 클라이머들의 생생한 후기를 확인해보세요.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <Card className="bg-gray-900 border-gray-800">
            <CardContent className="pt-6">
              <p className="italic text-gray-300">
                "혼자 클라이밍하면서 영상 촬영이 항상 고민이었는데, CruxCut 덕분에 고정 카메라로 찍어도 마치 누군가
                따라다니며 촬영한 것처럼 영상이 나와요. 움직임 분석도 정확해서 크럭스 구간을 놓치지 않고 잘 담아줍니다."
              </p>
            </CardContent>
            <CardFooter className="flex items-center gap-4 pt-6 border-t border-gray-800">
              <Avatar>
                <AvatarImage src="/placeholder.svg?height=40&width=40" />
                <AvatarFallback>JK</AvatarFallback>
              </Avatar>
              <div>
                <p className="font-medium">김정훈</p>
                <p className="text-sm text-gray-400">프로 클라이머</p>
              </div>
            </CardFooter>
          </Card>

          <Card className="bg-gray-900 border-gray-800">
            <CardContent className="pt-6">
              <p className="italic text-gray-300">
                "클라이밍 센터를 운영하면서 회원들의 클라이밍 영상을 제공하는 서비스를 시작했는데, CruxCut 덕분에 인력
                추가 없이 고품질 영상을 제공할 수 있게 되었습니다. 회원들의 만족도가 크게 올라갔어요."
              </p>
            </CardContent>
            <CardFooter className="flex items-center gap-4 pt-6 border-t border-gray-800">
              <Avatar>
                <AvatarImage src="/placeholder.svg?height=40&width=40" />
                <AvatarFallback>SJ</AvatarFallback>
              </Avatar>
              <div>
                <p className="font-medium">이수진</p>
                <p className="text-sm text-gray-400">클라이밍 센터 운영자</p>
              </div>
            </CardFooter>
          </Card>

          <Card className="bg-gray-900 border-gray-800">
            <CardContent className="pt-6">
              <p className="italic text-gray-300">
                "SNS에 클라이밍 영상을 올리는 것을 좋아하는데, CruxCut으로 편집한 영상은 조회수가 확실히 더 높아요.
                역동적인 앵글 전환과 하이라이트 편집 기능이 정말 마음에 듭니다. 이제 촬영자 없이도 멋진 영상을 만들 수
                있어요."
              </p>
            </CardContent>
            <CardFooter className="flex items-center gap-4 pt-6 border-t border-gray-800">
              <Avatar>
                <AvatarImage src="/placeholder.svg?height=40&width=40" />
                <AvatarFallback>MJ</AvatarFallback>
              </Avatar>
              <div>
                <p className="font-medium">박민준</p>
                <p className="text-sm text-gray-400">아마추어 클라이머</p>
              </div>
            </CardFooter>
          </Card>
        </div>
      </div>
    </section>
  )
}
